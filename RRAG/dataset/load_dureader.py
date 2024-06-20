import torch
import numpy as np
import random, os
def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


from transformers import AutoTokenizer
from typing import List, Optional, Tuple, Type, TypeVar
from copy import deepcopy
from pydantic.dataclasses import dataclass
import pathlib
import pickle
from tqdm import tqdm
import json, logging

T = TypeVar("T")

PROMPTS_ROOT = pathlib.Path('RRAG/prompts').resolve()


logger = logging.getLogger()
T = TypeVar("T")

def get_qa_instruction(
    question: str, documents: List, retrieval_aware: bool, RETRIEVAL_TOKEN, use_cot=False):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not documents:
        raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

    if similarity_aware:
        prompt_filename = 'qa_similarity_zh.prompt'
    else:
        prompt_filename = "qa_zh.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    
    formatted_documents = []
    for document_index, document in enumerate(documents):
        if retrieval_aware:
            document_prompt = f"[{document_index+1}]相关度: {RETRIEVAL_TOKEN}(标题：{document['title']}) {document['text']}"
        else:
            document_prompt = f"[{document_index+1}](标题：{document['title']}) {document['text']}"
        formatted_documents.append(document_prompt)
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def get_instruction_dataset(dataset, idx, max_prompt_length, tokenizer, retrieval_aware, RETRIEVAL_TOKEN, sample_answer=True):
    instruction_dataset = []
    for i in tqdm(idx):
        input_example = deepcopy(dataset[i])
        question = input_example["input"]
        documents = []
        for ctx in deepcopy(input_example["passages"]):
            documents.append(ctx)
        if not documents:
            raise ValueError(f"Did not find any documents for example: {input_example}")
        prompt = get_qa_instruction(
                question,
                documents,
                retrieval_aware=retrieval_aware,
                RETRIEVAL_TOKEN=RETRIEVAL_TOKEN,
            )
        
        input_example['instruction'] = prompt

        answers = random.sample(input_example['answers'], 1) if sample_answer else input_example['answers']
        for ans in answers:
            prompt_length = len(tokenizer(prompt + ans)["input_ids"])
            if max_prompt_length < prompt_length:
                logger.info(
                            f"Skipping prompt ... with length {prompt_length}, which "
                            f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            data = deepcopy(input_example)
            data['output'] = ans
            instruction_dataset.append(data)
    return instruction_dataset

def get_embeds(dataset):
    for data in dataset:
        data['embeds'] = [[float(d['rerank_score']), float(d['rerank_nb_score']), float(d['rerank_precedent_score'])] for d in data['passages']]
        data['label'] = [1 if d['is_selected'] else 0 for d in data['passages']]
    return dataset

def load_dureader_data(input_path, dataset_seed=42):
    with open(input_path, 'rb') as f:
        dureader_dataset = pickle.load(f)
        f.close()
    # dataset splitting
    seed_it(dataset_seed)
    all_index = list(range(len(examples)))
    train_index = np.sort(random.sample(all_index, int(len(all_index)*0.8)))
    test_index = np.sort(list(set(all_index).difference(set(train_index))))
    print(f'prepare dataset, train size: {len(train_index)}, test size: {len(test_index)}')
    return examples, train_index, test_index


def load_dureader_dataset(input_path, max_prompt_length, tokenizer, retrieval_aware=True, RETRIEVAL_TOKEN='<R>', dataset_seed=42):
    examples, train_index, test_index = load_nq_data(input_path, dataset_seed)
    instruction_dataset_train = get_instruction_dataset(examples, train_index, max_prompt_length, tokenizer, retrieval_aware, RETRIEVAL_TOKEN)
    instruction_dataset_test = get_instruction_dataset(examples, test_index, max_prompt_length, tokenizer, retrieval_aware, RETRIEVAL_TOKEN)

    instruction_dataset_train = get_embeds(instruction_dataset_train)
    instruction_dataset_test = get_embeds(instruction_dataset_test)
    return instruction_dataset_train, instruction_dataset_test

def get_dureader_ans(dataset):
    return [data['answers'] for data in dataset]


