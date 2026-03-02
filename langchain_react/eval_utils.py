import json
import re
import string
from collections import Counter

DATA_DIR = "../data"
HOTPOTQA_SPLIT_FILE = {
    "train": "hotpot_train_v1.1_simplified.json",
    "dev": "hotpot_dev_v1_simplified.json",
    "test": "hotpot_test_v1_simplified.json",
}


def load_hotpotqa(split="dev"):
    data_file = f"{DATA_DIR}/{HOTPOTQA_SPLIT_FILE[split]}"
    with open(data_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(d["question"], d["answer"]) for d in raw]


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def get_metrics(pred_answer, gt_answer):
    pred = normalize_answer(pred_answer)
    gt = normalize_answer(gt_answer)
    em = int(pred == gt)
    f1 = f1_score(pred_answer, gt_answer)[0]
    return {"em": em, "f1": f1, "reward": em}
