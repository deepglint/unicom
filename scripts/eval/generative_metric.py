import Levenshtein
import unicodedata
import json
import numpy as np
import re
import statistics
from typing import List
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu

# ---------------------------------- Metrics For OcrBench -----------------------------------
OCRBench_score = {
    "Regular Text Recognition": 0,
    "Irregular Text Recognition": 0,
    "Artistic Text Recognition": 0,
    "Handwriting Recognition": 0,
    "Digit String Recognition": 0,
    "Non-Semantic Text Recognition": 0,
    "Scene Text-centric VQA": 0,
    "Doc-oriented VQA": 0,
    "Key Information Extraction": 0,
    "Handwritten Mathematical Expression Recognition": 0,
}
def ocrbench_process_results(doc, results):
    pred = results[0].lower().strip()
    gt_ans = doc["answer"]
    dataset_name = doc["dataset"]

    score = 0
    if dataset_name == "HME100k":
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].strip().replace("\n", " ").replace(" ", "")
                predict = pred.strip().replace("\n", " ").replace(" ", "")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.strip().replace("\n", " ").replace(" ", "")
            predict = pred.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].lower().strip().replace("\n", " ")
                predict = pred.lower().strip().replace("\n", " ")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.lower().strip().replace("\n", " ")
            predict = pred.lower().strip().replace("\n", " ")
            if answer in predict:
                score = 1
    return {
        "ocrbench_accuracy": {"question_type": doc["question_type"], "score": score, "prediction": pred, "ground_truth": gt_ans},
    }

def get_ocrbench_score(contents_log):
    res_all = []
    for item in contents_log:
        res = ocrbench_process_results(item["doc"], item["resps"][0])
        res_all.append(res)
    res_all = [one["ocrbench_accuracy"]["score"]for one in res_all]
    return np.sum(res_all)/len(res_all)

def reformat_ocrbench_result(result_path):
    contents = json.loads(open(result_path).read())
    res_all = []
    for c in contents:
        elem = {
            "resps":[[c["pred"]]],
            "doc":{
                "dataset":c["type_level_2"],
                "question":c["question"],
                "question_type":c["type_level_1"],
                "answer":c["gt"]
            }
        }
        res_all.append(elem)
    return res_all

# ---------------------------------- Metrics For TextVQA -----------------------------------

class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (re.search(self.COMMA_STRIP, in_text) is not None):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item

def textvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    result = list(set(result))
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0
    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []
        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)
    return {
        "exact_match": accuracy
    }

def get_textvqa_score(contents_log):
    res_all = []
    for item in contents_log:
        res = textvqa_process_results(item["doc"], item["resps"][0])
        res_all.append(res)
    res_all = [one["exact_match"]for one in res_all]
    return np.mean(res_all)

def reformat_textvqa_result(result_path):
    contents = json.loads(open(result_path).read())
    res_all = []
    for c in contents:
        elem = {
            "resps":[[c["pred"]]],
            "doc":{
                "dataset":c["type_level_2"],
                "question":c["question"],
                "question_type":c["type_level_1"],
                "answers":c["gt"]
            }
        }
        res_all.append(elem)
    return res_all

# ---------------------------------- Metrics For DocVQA -----------------------------------

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
def anls(
    references,
    predictions,
    thresh_hold=0.5,
):  # This is a passthrough function
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(predictions[0].strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(predictions[0].upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < thresh_hold:
        question_result = 0
    return  question_result

def get_anls_score(answers:List, predictions:List):
    res_all = []
    for idx, elem in enumerate(answers):
        res = anls(elem, predictions[idx])
        res  = 0 if res < 0.5 else 1
        res_all.append(res)
    anls_socre = sum(res_all)/len(answers)
    return anls_socre

def get_docvqa_score(contents_log):
    references = [elem["doc"]["answers"] for elem in contents_log]
    predictions = [elem["resps"][0] for elem in contents_log]
    anls_score= get_anls_score(references, predictions)
    return anls_score

def reformat_docvqa_result(result_path):
    contents = json.loads(open(result_path).read())
    res_all = []
    for c in contents:
        elem = {
            "resps":[[c["pred"]]],
            "doc":{
                "dataset":c["type_level_2"],
                "question":c["question"],
                "question_type":c["type_level_1"],
                "answers":c["gt"]
            }
        }
        res_all.append(elem)
    return res_all

# ---------------------------------- Metrics For ChartQA -----------------------------------

def chartqa_process_results(doc, results):
    pred = results[0]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    return return_dict

def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def get_chartqa_score(contents_log):
    res_all = []
    for item in contents_log:
        res = chartqa_process_results(item["doc"], item["resps"][0])
        res_all.append(res)
    res_all = [one["relaxed_overall"]for one in res_all]
    return np.mean(res_all)

def reformat_chartqa_result(result_path):
    contents = json.loads(open(result_path).read())
    res_all = []
    for c in contents:
        elem = {
            "resps":[[c["pred"]]],
            "doc":{
                "dataset":c["type_level_2"],
                "question":c["question"],
                "question_type":c["type_level_1"],
                "answer":c["gt"][0]
            }
        }
        res_all.append(elem)
    return res_all

# ---------------------------------- Metrics For RoboVQA -----------------------------------

def robovqa_process_results(doc, results):
    pred = results.replace("\n", "").lower()
    gt = doc["answer"].replace("\n", "").lower()
    if gt in ['yes', 'no']:
        pred = re.sub(r'\b\w*yes\w*\b', 'yes', pred)
        pred = re.sub(r'\b\w*no\w*\b', 'no', pred)
    score, bleu1, bleu2, bleu3, bleu4 = get_bleu_score(pred, gt)
    return_dict = {
        "score": score, 
        "bleu1": bleu1, 
        "bleu2": bleu2, 
        "bleu3": bleu3, 
        "bleu4": bleu4
    }
    return return_dict

def get_bleu_score(prediction, target):
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    candidate = list(prediction.split(" "))
    reference = [list(target.split(" "))]
    if target is not None:
        # print(f"pred:{pred}, gt:{gt}, bleu:{sentence_bleu(reference, candidate)}")
        if len(reference[0]) <= 1:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
        elif len(reference[0]) == 2:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
        elif len(reference[0]) == 3:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00))
        else:
            bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00))
            bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00))
            bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00))
            bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
               
    score = (bleu1 + bleu2 + bleu3 + bleu4) / 4
    return score, bleu1, bleu2, bleu3, bleu4

def get_robovqa_score(contents_log):
    res_all = []
    for item in contents_log:
        res = robovqa_process_results(item["doc"], item["resps"])
        res_all.append(res)
    res_score = [one["score"] for one in res_all]
    res_bleu1 = [one["bleu1"] for one in res_all]
    res_bleu2 = [one["bleu2"] for one in res_all]
    res_bleu3 = [one["bleu3"] for one in res_all]
    res_bleu4 = [one["bleu4"] for one in res_all]
    return {
        "score": np.mean(res_score),
        "bleu1": np.mean(res_bleu1), 
        "bleu2": np.mean(res_bleu2), 
        "bleu3": np.mean(res_bleu3), 
        "bleu4": np.mean(res_bleu4)
    }

def reformat_robovqa_result(result_path):
    contents = json.loads(open(result_path).read())
    res_all = []
    for c in contents:
        elem = {
            "resps":c["pred"],
            "doc":{
                "dataset":c["type_level_2"],
                "question":c["question"],
                "question_type":c["type_level_1"],
                "answer":c["gt"][0]
            }
        }
        res_all.append(elem)
    return res_all

