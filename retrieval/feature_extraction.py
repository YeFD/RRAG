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
from xopen import xopen
from tqdm import tqdm
import json
import argparse
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import RerankingEvaluator
from tqdm import tqdm, trange
from retrieval_utils import *


### NQ-k ###
def load_data_nq(input_path, dataset_seed=42):
    examples = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            examples.append(deepcopy(input_example))
        fin.close()
    seed_it(dataset_seed)
    all_index = list(range(len(examples)))
    train_index = np.sort(random.sample(all_index, int(len(all_index)*0.8)))
    test_index = np.sort(list(set(all_index).difference(set(train_index))))
    print(f'prepare dataset, train size: {len(train_index)}, test size: {len(test_index)}')
    return examples, all_index, train_index, test_index

### HotpotQA ### ### 2Wiki ###
def load_data_json(input_path, dataset_seed=42):
    train_data_path = input_path['train_data_path']
    test_data_path = input_path['test_data_path']
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        f.close()
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        f.close()
    print(f'prepare dataset, train size: {len(train_data)}, test size: {len(test_data)}')
    return train_data, test_data

### MuSiQue ###
def load_data_jsonl(input_path, dataset_seed=42):
    train_data_path = input_path['train_data_path']
    test_data_path = input_path['test_data_path']
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = []
        for line in f.readlines():
            data = json.loads(line)
            if len(data['paragraphs']) != 20:
                continue
            train_data.append(data)
        f.close()
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = []
        for line in f.readlines():
            data = json.loads(line)
            if len(data['paragraphs']) != 20:
                continue
            test_data.append(data)
        f.close()
    print(f'prepare dataset, train size: {len(train_data)}, test size: {len(test_data)}')
    return train_data, test_data

##### Feature Extration and Save
def main(dataset_name, input_path, train_data_path, test_data_path, model_name, save_path, save_train_path, save_test_path, output_path):
    ##### Load Retriever
    model = SentenceTransformer(model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ### NQ ###
    if 'nq' in dataset_name or dataset_name == 'dureader':
        examples, all_index, train_index, test_index = load_data_nq(input_path)
        ##### Evaluation
        test_samples = get_dual_dev(examples, test_index)

        dev_evaluator = RerankingEvaluator(test_samples, batch_size=32, show_progress_bar=True)
        r = dev_evaluator(model, output_path) 

        dataset = get_dual_sim(examples, all_index, model)
        with open(save_path, 'wb') as fin:
            pickle.dump(dataset, fin)
            fin.close()
    ### HotpotQA ### ### 2Wiki ###
    elif dataset_name == 'hotpotqa' or dataset_name == '2wiki':
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
        train_data, test_data = load_data_json(input_path)
        test_samples = get_dual_dev_hotpotqa(test_data)

        dev_evaluator = RerankingEvaluator(test_samples, batch_size=32, show_progress_bar=True)
        r = dev_evaluator(model, output_path) 

        dataset = get_dual_sim_hotpotqa(train_data, model)
        dataset_test = get_dual_sim_hotpotqa(test_data, model)
        with open(save_train_path, 'wb') as fin:
            pickle.dump(dataset, fin)
            fin.close()
        with open(save_test_path, 'wb') as fin:
            pickle.dump(dataset_test, fin)
            fin.close()
    ### MuSiQue ###
    elif dataset_name == 'musique':
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
        train_data, test_data = load_data_jsonl(input_path)
        test_samples = get_dual_dev_musique(test_data)

        dev_evaluator = RerankingEvaluator(test_samples, batch_size=32, show_progress_bar=True)
        r = dev_evaluator(model, output_path) 

        dataset = get_dual_sim_musique(train_data, model)
        dataset_test = get_dual_sim_musique(test_data, model)
        with open(save_train_path, 'wb') as fin:
            pickle.dump(dataset, fin)
            fin.close()
        with open(save_test_path, 'wb') as fin:
            pickle.dump(dataset_test, fin)
            fin.close()
    else:
        raise ValueError(dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval feature extraction")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['nq_10', 'nq_20', 'nq_30', 'hotpotqa', 'musique', '2wiki', 'dureader'], help='Name of the dataset to process')    
    parser.add_argument('--input_path', type=str, required=False, default=None, help='Path for nq or dureader datasets')
    parser.add_argument('--train_data_path', type=str, required=False, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--test_data_path', type=str, required=False, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--model_name', type=str, required=True, help='SentenceTransformer model name or path for retriever')
    parser.add_argument('--save_path', type=str, required=False, help='Path to save train dataset')
    parser.add_argument('--save_train_path', type=str, required=False, help='Path to save train dataset')
    parser.add_argument('--save_test_path', type=str, required=False, help='Path to save test dataset')
    parser.add_argument('--output_path', type=str, required=False, default='temp_result', help='Path to save the evluation results')
    args = parser.parse_args()

    main(args.dataset_name, args.input_path, args.train_data_path, args.test_data_path, args.model_name, args.save_path, args.save_train_path, args.save_test_path, args.output_path)
    
