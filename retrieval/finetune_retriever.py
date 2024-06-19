# Copy from https://github.com/UKPLab/sentence-transformers/tree/master/examples/training
import torch
import numpy as np
import random, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

from xopen import xopen
from tqdm import tqdm
import json
import argparse
import math
from copy import deepcopy
from sentence_transformers import SentenceTransformer, util, InputExample, models
from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.losses import ContrastiveLoss
from datetime import datetime
from torch.utils.data import DataLoader
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
    train_samples = init_dual_samples(examples, train_index)
    test_samples = get_dual_dev(examples, test_index)
    return train_samples, test_samples

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
    train_samples = init_dual_samples_hotpotqa(train_data)
    test_samples = get_dual_dev_hotpotqa(dev_data)
    return train_samples, test_samples

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
    train_samples = init_dual_samples_musique(train_data[:])
    test_samples = get_dual_dev_musique(dev_data)
    return train_samples, test_samples
    

def main(dataset_name, input_path, train_data_path, test_data_path, model_name, model_save_path, train_batch_size, num_epochs):
    ### NQ ###
    if 'nq' in dataset_name or dataset_name == 'dureader':
        train_samples, test_samples = load_data_nq(input_path)
    ### HotpotQA ### ### 2Wiki ###
    elif dataset_name == 'hotpotqa' or dataset_name == '2wiki':
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
        train_samples, test_samples = load_data_json(input_path)
    ### MuSiQue ###
    elif dataset_name == 'musique':
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
        train_samples, test_samples = load_data_jsonl(input_path)
    print('len(train_samples), len(test_samples):', len(train_samples), len(test_samples))

    ##### Load Retriever
    max_seq_length = 512
    train_batch_size = int(train_batch_size)
    num_epochs = int(num_epochs)
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model_save_path = 'retrievers/bert'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if model_save_path is None else model_save_path
    print('model_save_path', model_save_path)

    ##### Init
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
    train_loss = ContrastiveLoss(model)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    evaluation_steps = math.ceil(len(train_dataloader) * 0.5) #Evaluate every 50% of the data#40

    ##### Eval
    dev_evaluator = RerankingEvaluator(test_samples, batch_size=64, show_progress_bar=True)
    m = dev_evaluator(model)

    ##### Train
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_params={'lr': 3e-5},
            use_amp=True,        #Set to True, if your GPU supports FP16 cores
    )

    ##### Eval
    r = dev_evaluator(model, output_path='temp_result') # bert finetune 0.9601560398170568

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a retriever")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['nq_10', 'nq_20', 'nq_30', 'hotpotqa', 'musique', '2wiki', 'dureader'], help='Name of the dataset to process')    
    parser.add_argument('--input_path', type=str, required=False, default=None, help='Path for nq or dureader datasets')
    parser.add_argument('--train_data_path', type=str, required=False, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--test_data_path', type=str, required=False, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--model_name', type=str, required=True, help='SentenceTransformer model name or path for retriever')
    parser.add_argument('--model_save_path', type=str, required=False, help='Path to save retriever')
    parser.add_argument('--train_batch_size', type=int, required=False, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, required=False, default=4, help='Epoch')

    args = parser.parse_args()
    main(args.dataset_name, args.input_path, args.train_data_path, args.test_data_path, args.model_name, args.model_save_path, args.train_batch_size, args.num_epochs)

