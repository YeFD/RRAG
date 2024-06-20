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
T = TypeVar("T")
PROMPTS_ROOT = pathlib.Path('RRAG/prompts').resolve()

@dataclass(frozen=True)
class Document:
    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        rerank_score = data.pop("rerank_score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        if rerank_score is not None:
            rerank_score = float(rerank_score)
        return cls(**dict(data, id=id, score=score, rerank_score=rerank_score))

def get_qa_instruction(
    question: str, context: List, retrieval_aware: bool, RETRIEVAL_TOKEN, use_cot=False):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not context:
        raise ValueError(f"Provided `context` must be truthy, got: {context}")

    if retrieval_aware:
        prompt_filename = 'qa_similarity.prompt'
    elif use_cot:
        prompt_filename = 'qa_cot.prompt'
    else:
        prompt_filename = "qa.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    
    formatted_documents = []
    for document_index, document in enumerate(context):
        document_text = document['text']
        if retrieval_aware:
            document_prompt = f"[{document_index+1}]similarity: {RETRIEVAL_TOKEN}(Title: {document['title']}) {document_text}"
        else:
            document_prompt = f"[{document_index+1}](Title: {document['title']}) {document_text}"
        formatted_documents.append(document_prompt)
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def get_instruction_dataset(dataset, max_prompt_length, tokenizer, retrieval_aware, RETRIEVAL_TOKEN, sample_answer=True):
    instruction_dataset = []
    for input_example in tqdm(dataset):
        input_example = deepcopy(input_example)
        question = input_example["question"]
        context = input_example["context"]
        if not context:
            raise ValueError(f"Did not find any documents for example: {input_example}")
        prompt = get_qa_instruction(
                question,
                context,
                retrieval_aware=retrieval_aware,
                RETRIEVAL_TOKEN=RETRIEVAL_TOKEN,
            )
        
        input_example['instruction'] = prompt

        answers = [input_example['answer']]
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
        supporting_facts = data['supporting_facts']
        data['embeds'] = [[float(d['rerank_score']), float(d['rerank_nb_score']), float(d['rerank_precedent_score'])] for d in data['context']]
        data['label'] = [1 if d['title'] in supporting_facts else 0 for d in data['context']]
    return dataset

def pre_hotpotqa(dataset):
    remove_num = 0
    dataset_new = []
    for data in dataset:
        data = deepcopy(data)
        supporting_facts = [d[0] for d in data['supporting_facts']]
        if len(data['context']) != 10:
            remove_num += 1
            continue
        context = [
            {
                'title': d[0], 
                'text': ''.join(d[1]),
                'rerank_score': float(d[2][0]),
                'rerank_nb_score': float(d[2][1]),
                'rerank_precedent_score': float(d[2][2]),
                'is_supporting': 1 if d[0] in supporting_facts else 0,
                } 
            for d in data['context']]
        data['supporting_facts'] = supporting_facts
        data['context'] = context
        dataset_new.append(data)
    print('remove_num', remove_num)
    return dataset_new

def load_hotpotqa_data(input_path):
    train_data_path = input_path['train_data_path']
    test_data_path = input_path['test_data_path']
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)[:]
        f.close()
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)[:]
        f.close()
    train_data = pre_hotpotqa(train_data[:])
    test_data = pre_hotpotqa(test_data[:])
    print(f'prepare dataset, train size: {len(train_data)}, test size: {len(test_data)}')
    return train_data, test_data


def load_hotpotqa_dataset(input_path, max_prompt_length, tokenizer, retrieval_aware=True, RETRIEVAL_TOKEN='<R>'):
    train_data, test_data = load_hotpotqa_data(input_path)
    instruction_dataset_train = get_instruction_dataset(train_data[:], max_prompt_length, tokenizer, retrieval_aware, RETRIEVAL_TOKEN)
    instruction_dataset_test = get_instruction_dataset(test_data[:], max_prompt_length, tokenizer, retrieval_aware, RETRIEVAL_TOKEN)

    instruction_dataset_train = get_embeds(instruction_dataset_train)
    instruction_dataset_test = get_embeds(instruction_dataset_test)
    return instruction_dataset_train, instruction_dataset_test

def get_hotpotqa_ans(dataset):
    return [[data['answer']] for data in dataset]


