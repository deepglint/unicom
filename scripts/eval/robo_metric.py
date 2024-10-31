import json
import numpy as np
import re
from collections import defaultdict
from tqdm import tqdm
import sys, os
from third_party.openeqa.evaluation.llm_match import get_llm_match_score
from nltk.translate.bleu_score import sentence_bleu

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
                "answer":c["gt"]
            }
        }
        res_all.append(elem)
    return res_all

def get_robovqa_score(result_path):
    
    contents = reformat_robovqa_result(result_path)
    res_all = []

    for item in contents:
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
    }, contents

# ---------------------------------- Metrics For OpenEQA -----------------------------------

def get_output_filename(path):
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    name, ext = os.path.splitext(base_name)
    new_filename = f"data_with_llm_score{ext}"
    new_path = os.path.join(dir_name, new_filename)
    return new_path

def req_openeqa_llm_score(result_path):

    with open(result_path, "r") as f:
        contents = json.load(f)

    output_file = get_output_filename(result_path)
    temp_data, error_data = [], []
    resume_flag = False

    try:
        with open(output_file, "r") as f:
            temp_data = json.load(f)
            resume_id = temp_data[-1]["unique_id"]
    except FileNotFoundError:
        resume_id = -1
        resume_flag = True
        print("No json file to be loaded.")
    
    for i, item in enumerate(tqdm(contents)):
        
        if item["unique_id"] == resume_id:
            resume_flag = True      
            continue              
        
        if resume_flag:
            question = item["question"]
            answer = item["gt"]
            extra_answers = item["extra_gt"]
            prediction = item["pred"]
            score = get_llm_match_score(question, answer, prediction, extra_answers=extra_answers, endpoint=True)
            if isinstance(score, str):
                print("*" * 40)
                print("Item ID:    {}".format(i))
                print("Example question:    {}".format(question))
                print("Ground-truth extra-answers: {}".format(extra_answers))
                print("Ground-truth answer: {}".format(answer))
                print("Predicted answer:    {}".format(prediction))
                print(f"Saving temporary result into {output_file}")
                error_data.append(item)
            else:
                llm_score = (score - 1) / 4
                item["llm_score"] = llm_score   

                print("*" * 40)
                print("Item ID:    {}".format(i))
                print("Example question:    {}".format(question))
                print("Ground-truth extra-answers: {}".format(extra_answers))
                print("Ground-truth answer: {}".format(answer))
                print("Predicted answer:    {}".format(prediction))
                print("LLM-match score:     {}".format(llm_score))

                temp_data.append(item)

                if (i + 1) % 50 == 0 or (i + 1) == len(contents):
                    print(f"Saving temporary result into {output_file}")
                    with open(output_file, 'w') as outfile:
                        json.dump(temp_data, outfile, indent=4)

    if len(error_data) > 0:
        print("*" * 40)
        dir_path = os.path.dirname(output_file)
        error_path = os.path.join(dir_path, 'score_error.json')
        with open(error_path, 'a') as outfile:
            json.dump(error_data, outfile, indent=4)
        print(f"Saving error response log into {error_path}")

    return output_file
    
def get_openeqa_score(result_path):
    
    scores = defaultdict(list)
    scores_avg = defaultdict(float)
    output_file = req_openeqa_llm_score(result_path)

    with open(output_file, "r") as f:
        contents = json.load(f)

    for item in contents:
        type_level_1 = item['type_level_1']
        llm_score = item['llm_score']
        scores[type_level_1].append(llm_score)
        
    for item in contents:
        type_level_2 = item['type_level_2']
        llm_score = item['llm_score']
        scores[type_level_2].append(llm_score)

    for type_name, score_list in scores.items():
        average_score = sum(score_list) / len(score_list)
        scores_avg[type_name] = round(average_score, 4)
    
    return scores_avg, contents




