import torch
import numpy as np
import random, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
print(torch.cuda.is_available())
def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
seed_it(42)

import pickle
import json
import httpx
import logging
import argparse
from xopen import xopen
from copy import deepcopy
from tqdm import tqdm, trange
from copy import deepcopy
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics import average_precision_score, ndcg_score
from typing import Callable, Optional
from openai import OpenAI

from retrieval_utils import get_precedent_sim, get_nb_sim

def getClient()->OpenAI:
    client = OpenAI(
        base_url="https://api/v1", 
        api_key="sk-xxx",
    )
    return client
client = getClient()

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_sample_embedding(data, model):
    result = {}
    query = data['question']
    query_embeds = get_embedding(query, model)
    result['query'] = query
    result['query_embeds'] = query_embeds
    result['ctxs'] = []
    for ctx in data['ctxs']:
        text = 'Title: '+ctx['title'] +'\n' + ctx['text']
        embeds = get_embedding(text)
        result['ctxs'].append({'text': text, 'embeds': embeds})
    return result

def get_dual_sim(dataset, idx, dataset_embeds):
    dataset_new = []
    for i in tqdm(idx):
        data = deepcopy(dataset[i])
        query = data['question']
        ctxs_text = []
        for ctx in data['ctxs']:
            text = 'Title: '+ctx['title'] +'\n' + ctx['text']
            ctxs_text.append(text)
        data_embeds = dataset_embeds[i]
        q_emb = torch.Tensor(data_embeds['query_embeds']).cuda()
        c_emb = torch.Tensor([ctx['embeds'] for ctx in data_embeds['ctxs']]).cuda()
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

logger = logging.getLogger(__name__)

##### Evaluator for Embeddings
class RerankingEvaluator(SentenceEvaluator):
    def __init__(
        self,
        at_k: int = 10,
        write_csv: bool = True,
        similarity_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cos_sim,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        use_batched_encoding: bool = True,
        truncate_dim: Optional[int] = None,
        mrr_at_k: Optional[int] = None,
    ):

        if mrr_at_k is not None:
            logger.warning(f"The `mrr_at_k` parameter has been deprecated; please use `at_k={mrr_at_k}` instead.")
            self.at_k = mrr_at_k
        else:
            self.at_k = at_k
        self.similarity_fct = similarity_fct
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.use_batched_encoding = use_batched_encoding
        self.truncate_dim = truncate_dim

    def compute_metrices_from_embeds(self, dataset, dataset_embeds):
        """
        Embeds every (query, positive, negative) tuple individually.
        Is slower than the batched version, but saves memory as only the
        embeddings for one tuple are needed. Useful when you have
        a really large test set
        """
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []
        if len(dataset) != len(dataset_embeds):
            raise ValueError()

        for idx in trange(len(dataset), desc="Samples"):
            data = dataset[idx]
            ctxs = data['ctxs']
            is_relevant = [1 if d['isgold'] else 0 for d in data['ctxs']]
            instance = dataset_embeds[idx]
            query_embeds = instance["query_embeds"]
            ctxs_embeds = [ctx['embeds'] for ctx in instance['ctxs']]

            if sum(is_relevant) == 0 or sum(is_relevant) == len(is_relevant):
                raise ValueError()

            query_emb = torch.Tensor(query_embeds).cuda()
            docs_emb = torch.Tensor(ctxs_embeds).cuda()

            pred_scores = self.similarity_fct(query_emb, docs_emb)
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]

            pred_scores_argsort = torch.argsort(-pred_scores)  # Sort in decreasing order
            pred_scores = pred_scores.cpu().tolist()

            # Compute MRR score
            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0 : self.at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank + 1)
                    break
            all_mrr_scores.append(mrr_score)

            # Compute NDCG score
            all_ndcg_scores.append(ndcg_score([is_relevant], [pred_scores], k=self.at_k))

            # Compute AP
            all_ap_scores.append(average_precision_score(is_relevant, pred_scores))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)

        return {"map": mean_ap, "mrr": mean_mrr, "ndcg": mean_ndcg}

def main(dataset_name, input_path, model_name, emb_save_path, dataset_save_path, dataset_seed=42):
    ##### Load Data
    # NQ-k datasets download from https://github.com/nelson-liu/lost-in-the-middle/tree/main/qa_data
    examples = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            examples.append(deepcopy(input_example))
        fin.close()
    seed_it(dataset_seed)
    all_index = list(range(len(examples)))

    ##### Get Embeddings
    dataset_embedding = []
    for i in tqdm(all_index[:]):
        data = deepcopy(examples[i])
        result = get_sample_embedding(data, model_name)
        dataset_embedding.append(result)
    with open(emb_save_path, 'wb') as f:
        pickle.dump(dataset_embedding, f)
        f.close()

    ##### Evaluation
    evaluator = RerankingEvaluator()
    print(evaluator.compute_metrices_from_embeds(examples[:], dataset_embedding))

    ##### Feature Extration
    res = get_dual_sim(examples, all_index[:], dataset_embedding)

    ##### Save Dataset
    with open(dataset_save_path, 'wb') as fin:
        pickle.dump(res, fin)
        fin.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval feature extraction with OpenAI Embedding")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['nq_10', 'nq_20', 'nq_30',], help='Name of the dataset to process')    
    parser.add_argument('--input_path', type=str, required=False, default=None, help='Path for nq datasets')
    parser.add_argument('--model_name', type=str, required=True, help='OpenAI Embedding model name')
    parser.add_argument('--emb_save_path', type=str, required=False, help='Path to save embeddings')
    parser.add_argument('--dataset_save_path', type=str, required=False, help='Path to save train dataset')

    args = parser.parse_args()
    main(args.dataset_name, args.input_path, args.model_name, args.emb_save_path, args.dataset_save_path)

