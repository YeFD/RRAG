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

logger = logging.getLogger()
PROMPTS_ROOT = pathlib.Path('RRAG/prompts').resolve()
T = TypeVar("T")

def get_qa_instruction(
    question: str, paragraphs: List, retrieval_aware: bool, RETRIEVAL_TOKEN, use_cot=False):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not paragraphs:
        raise ValueError(f"Provided `paragraphs` must be truthy, got: {paragraphs}")

    if retrieval_aware:
        prompt_filename = 'qa_similarity.prompt'
    elif use_cot:
        prompt_filename = 'qa_cot.prompt'
    else:
        prompt_filename = "qa.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    
    formatted_documents = []
    for document_index, document in enumerate(paragraphs):
        if retrieval_aware:
            document_prompt = f"[{document_index+1}]similarity: {RETRIEVAL_TOKEN}(Title: {document['title']}) {document['paragraph_text']}"
        else:
            document_prompt = f"[{document_index+1}](Title: {document['title']}) {document['paragraph_text']}"
        formatted_documents.append(document_prompt)
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def get_instruction_dataset(dataset, max_prompt_length, tokenizer, retrieval_aware, RETRIEVAL_TOKEN, sample_answer=True):
    instruction_dataset = []
    for input_example in tqdm(dataset):
        input_example = deepcopy(input_example)
        question = input_example["question"]
        paragraphs = input_example["paragraphs"]
        if not paragraphs:
            raise ValueError(f"Did not find any documents for example: {input_example}")
        prompt = get_qa_instruction(
                question,
                paragraphs,
                retrieval_aware=retrieval_aware,
                RETRIEVAL_TOKEN=RETRIEVAL_TOKEN,
            )
        
        input_example['instruction'] = prompt

        answers = [input_example['answer']] if sample_answer else [input_example['answer']] + input_example['answer_aliases']
        for ans in answers:
            prompt_length = len(tokenizer(prompt + ans)["input_ids"])
            if max_prompt_length < prompt_length:
                print(
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
        data['embeds'] = [[float(d['rerank_score']), float(d['rerank_nb_score']), float(d['rerank_precedent_score'])] for d in data['paragraphs']]
        data['label'] = [1 if d['is_supporting'] else 0 for d in data['paragraphs']]
    return dataset

def load_musique_data(input_path):
    train_data_path = input_path['train_data_path']
    test_data_path = input_path['test_data_path']
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)[:]
        f.close()
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)[:]
        f.close()
    print(f'prepare dataset, train size: {len(train_data)}, test size: {len(test_data)}')
    return train_data, test_data


def load_musique_dataset(input_path, max_prompt_length, tokenizer, retrieval_aware=True, RETRIEVAL_TOKEN='<R>'):
    train_data, test_data = load_musique_data(input_path)
    instruction_dataset_train = get_instruction_dataset(train_data[:], max_prompt_length, tokenizer, retrieval_aware, RETRIEVAL_TOKEN)
    instruction_dataset_test = get_instruction_dataset(test_data[:], max_prompt_length, tokenizer, retrieval_aware, RETRIEVAL_TOKEN)

    instruction_dataset_train = get_embeds(instruction_dataset_train)
    instruction_dataset_test = get_embeds(instruction_dataset_test)
    return instruction_dataset_train, instruction_dataset_test

def get_musique_ans(dataset):
    return [[data['answer']]+data['answer_aliases'] for data in dataset]


