# copy from https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py and https://github.com/THUDM/LongBench/blob/main/metrics.py
import argparse
import json
import logging
import statistics
import sys
import string
import jieba
import regex
from copy import deepcopy
from tqdm import tqdm, trange
from xopen import xopen
from typing import List
from rouge import Rouge

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

from collections import Counter
def f1_score(prediction: str, ground_truth: List[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truths):
    normalized_prediction = normalize_answer(prediction)
    score = []
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        score.append(f1_score(prediction_tokens, ground_truth_tokens))
    return max(score)


logger = logging.getLogger(__name__)

def normalize_zh_answer(s):
    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def best_subspan_em_zh(prediction: str, ground_truths: List[str]) -> float:
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    for ground_truth in ground_truths:
        ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
        ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
        ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
        if ''.join(prediction_tokens) in ''.join(ground_truth_tokens):
            return 1.0
    return 0.0

def qa_f1_zh_score(prediction, ground_truths, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    score = []
    for ground_truth in ground_truths:
        ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
        ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
        ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
        score.append(f1_score(prediction_tokens, ground_truth_tokens))
    return max(score)

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truths):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    score = []
    for ground_truth in ground_truths:
        ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
        score.append(rouge_score(prediction, ground_truth))
    return max(score)


def get_metrics_for_example_zh(example):
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    # model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)

def get_metrics_for_example(example, METRICS):
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)

def evaluation_from_list(responses, answers, dataset_name):
    if 'nq' in dataset_name:
        METRICS = METRICS_NQ
    elif 'dureader' in dataset_name:
        METRICS = METRICS_ZH
    else:
        METRICS = METRICS_EN

    logger.info("Computing metrics")
    all_example_metrics = []
    if len(responses) != len(answers):
        raise ValueError
    for i in trange(len(responses)):
        all_example_metrics.append(get_metrics_for_example({'model_answer': responses[i], 'answers': answers[i]}, METRICS))

    # Average metrics across examples

    for (_, metric_name) in METRICS:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        logger.info(f"{metric_name}: {average_metric_value}")
        print(f"{metric_name}: {average_metric_value}")
    return all_example_metrics

METRICS_NQ = [
    (best_subspan_em, "best_subspan_em"),
]
METRICS_EN = [
    (best_subspan_em, "best_subspan_em"),
    (qa_f1_score, "qa_f1_score"),
]
METRICS_ZH = [
    (qa_f1_zh_score, "qa_f1_zh_score"),
    (rouge_zh_score, "rouge_zh_score")
]
