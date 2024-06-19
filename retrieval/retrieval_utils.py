import torch
from tqdm import tqdm
from sentence_transformers.util import cos_sim
from copy import deepcopy
import numpy as np
from sentence_transformers import SentenceTransformer, util, InputExample, models

def get_precedent_sim(q_emb, c_emb):
    cosine_score = cos_sim(q_emb, c_emb)[0].cpu()
    rank_list = np.array(cosine_score).argsort().tolist()[::-1]
    precedent_sim = [1]
    for i in range(1, len(rank_list)):
        precedent_score = np.array([cosine_score[idx] for idx in rank_list[:i]])
        precedent_weight = np.exp(precedent_score)/np.exp(precedent_score).sum()
        w = torch.Tensor(precedent_weight, device='cpu').reshape(1, precedent_score.shape[0])
        precedent_embs = [c_emb[idx].reshape(1, c_emb.shape[1]).cpu() for idx in rank_list[:i]]
        precedent_emb = torch.mm(w, torch.cat(precedent_embs))
        cur_emb = c_emb[rank_list[i]].cpu()
        score = cos_sim(cur_emb, precedent_emb)[0][0]
        precedent_sim.append(score)
    return np.array(cosine_score), rank_list, np.array(precedent_sim)

def get_nb_sim(c_emb, rank_list):
    rank_emb = [c_emb[i] for i in rank_list]
    nb_sim = [cos_sim(rank_emb[0], rank_emb[1]).cpu()]
    for i in range(1, len(rank_list)-1):
        nb_sim.append((cos_sim(rank_emb[i], rank_emb[i-1]).cpu() + cos_sim(rank_emb[i], rank_emb[i+1]).cpu()) / 2)
    nb_sim.append(cos_sim(rank_emb[-2], rank_emb[-1]).cpu())
    nb_sim = [s.squeeze() for s in nb_sim]
    return nb_sim


##### NQ datasets
def get_dual_sample_pair(query_text, ctxs, labels):
    samples = []
    for i in range(len(ctxs)):
        ctx = ctxs[i]
        text = 'Title: '+ctx['title'] +'\n' + ctx['text']
        sample = InputExample(texts=[query_text, text], label=1 if labels[i] else 0)
        samples.append(sample)
    return samples

def init_dual_samples(dataset, idx):
    samples = []
    for i in tqdm(idx):
        query_text = dataset[i]['question']
        ctxs = dataset[i]['ctxs']
        labels = [l['isgold'] for l in ctxs]
        samples = samples + get_dual_sample_pair(query_text, ctxs, labels)
    return samples

def get_dual_sim(dataset, idx, retrival_model):
    dataset_new = []
    for i in tqdm(idx):
        data = deepcopy(dataset[i])
        query = data['question']
        ctxs_text = []
        for ctx in data['ctxs']:
            text = 'Title: '+ctx['title'] +'\n' + ctx['text']
            ctxs_text.append(text)
        embs = retrival_model.encode([query] + ctxs_text, convert_to_tensor=True, normalize_embeddings=True)
        q_emb = embs[0]
        c_emb = embs[1:]
        scores, rank_list, precendent_scores = get_precedent_sim(q_emb, c_emb)
        nb_scores = get_nb_sim(c_emb, rank_list)
        ctxs = []
        for i in range(len(rank_list)):
            j = rank_list[i]
            ctx = deepcopy(data['ctxs'][j])
            ctx['rerank_score'] = float(scores[j])
            ctx['rerank_nb_score'] = float(nb_scores[i])
            ctx['rerank_precedent_score'] = float(precendent_scores[i])
            ctxs.append(ctx)
        data['ctxs'] = ctxs
        dataset_new.append(data)
    return dataset_new

def get_dual_dev(dataset, idx):
    dev_data = []
    for i in idx:
        data = dataset[i]
        query = data['question']
        pos = []
        neg = []
        for ctx in data['ctxs']:
            text = 'Title: '+ctx['title'] +'\n' + ctx['text']
            if ctx['isgold'] == 1:
                pos.append(text)
            else:
                neg.append(text)
        dev_data.append({'query': query, 'positive': pos, 'negative': neg})
    return dev_data

##### HotpotQA / 2Wiki
def get_dual_sample_pair_hotpotqa(query_text, context, supporting_facts):
    samples = []
    gt_titles = [s[0] for s in supporting_facts]
    for i in range(len(context)):
        ctx = context[i]
        text = 'Title: '+ctx[0] +'\n' + ''.join(ctx[1])
        label = 1 if ctx[0] in gt_titles else 0
        sample = InputExample(texts=[query_text, text], label=label)
        samples.append(sample)
    return samples

def init_dual_samples_hotpotqa(dataset):
    samples = []
    for data in tqdm(dataset):
        query_text = data['question']
        context = data['context']
        supporting_facts = data['supporting_facts']
        samples = samples + get_dual_sample_pair_hotpotqa(query_text, context, supporting_facts)
    return samples

def get_dual_dev_hotpotqa(dataset):
    dev_data = []
    for data in dataset:
        query = data['question']
        context = data['context']
        supporting_facts = data['supporting_facts']
        gt_titles = [s[0] for s in supporting_facts]
        pos = []
        neg = []
        for ctx in context:
            text = 'Title: '+ctx[0] +'\n' + ''.join(ctx[1])
            if ctx[0] in gt_titles:
                pos.append(text)
            else:
                neg.append(text)
        dev_data.append({'query': query, 'positive': pos, 'negative': neg})
    return dev_data

def get_dual_sim_hotpotqa(dataset, retrival_model):
    dataset_new = []
    for data in tqdm(dataset):
        data = deepcopy(data)
        query = data['question']
        context = data['context']
        supporting_facts = data['supporting_facts']
        gt_titles = [s[0] for s in supporting_facts]
        ctxs_text = []
        for ctx in context:
            text = 'Title: '+ctx[0] +'\n' + ''.join(ctx[1])
            ctxs_text.append(text)
        embs = retrival_model.encode([query] + ctxs_text, convert_to_tensor=True)
        q_emb = embs[0]
        c_emb = embs[1:]
        scores, rank_list, precendent_scores = get_precedent_sim(q_emb, c_emb)
        nb_scores = get_nb_sim(c_emb, rank_list)
        ctxs = []
        for i in range(len(rank_list)):
            j = rank_list[i]
            ctx = deepcopy(data['context'][j])
            rerank_score = float(scores[j])
            rerank_nb_score = float(nb_scores[i])
            rerank_precedent_score = float(precendent_scores[i])
            ctx.append((rerank_score, rerank_nb_score, rerank_precedent_score))
            ctxs.append(ctx)
        data['context'] = ctxs
        dataset_new.append(data)
    return dataset_new

##### MuSiQue
def get_dual_sample_pair_musique(query_text, paragraphs):
    samples = []
    for i in range(len(paragraphs)):
        ctx = paragraphs[i]
        text = 'Title: '+ctx['title'] +'\n' + ctx['paragraph_text']
        label = 1 if ctx['is_supporting'] else 0
        sample = InputExample(texts=[query_text, text], label=label)
        samples.append(sample)
    return samples

def init_dual_samples_musique(dataset):
    samples = []
    for data in tqdm(dataset):
        query_text = data['question']
        paragraphs = data['paragraphs']
        samples = samples + get_dual_sample_pair_musique(query_text, paragraphs)
    return samples

def get_dual_dev_musique(dataset):
    dev_data = []
    for data in dataset:
        query = data['question']
        paragraphs = data['paragraphs']
        pos = []
        neg = []
        for ctx in paragraphs:
            text = 'Title: '+ctx['title'] +'\n' + ctx['paragraph_text']
            if ctx['is_supporting']:
                pos.append(text)
            else:
                neg.append(text)
        dev_data.append({'query': query, 'positive': pos, 'negative': neg})
    return dev_data

def get_dual_sim_musique(dataset, retrival_model):
    dataset_new = []
    for data in tqdm(dataset):
        query = data['question']
        paragraphs = data['paragraphs']
        ctxs_text = []
        for ctx in paragraphs:
            text = 'Title: '+ctx['title'] +'\n' + ctx['paragraph_text']
            ctxs_text.append(text)
        embs = retrival_model.encode([query] + ctxs_text, convert_to_tensor=True)
        q_emb = embs[0]
        c_emb = embs[1:]
        scores, rank_list, precendent_scores = get_precedent_sim(q_emb, c_emb)
        nb_scores = get_nb_sim(c_emb, rank_list)
        ctxs = []
        for i in range(len(rank_list)):
            j = rank_list[i]
            ctx = deepcopy(data['paragraphs'][j])
            ctx['rerank_score'] = float(scores[j])
            ctx['rerank_nb_score'] = float(nb_scores[i])
            ctx['rerank_precedent_score'] = float(precendent_scores[i])
            ctxs.append(ctx)
        data['paragraphs'] = ctxs
        dataset_new.append(data)
    return dataset_new