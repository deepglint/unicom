import re
# from rouge import Rouge
import argparse
import os
import json
import numpy as np
from pycocotools import mask as mask_util
import numpy as np
import cv2
import base64
import urllib.request

spot_the_diff = ["Spot-the-Diff", "Birds-to-Words", "CLEVR-Change"]
image_edit_instruct = ["IEdit", "HQ-Edit", "MagicBrush"]
visual_story_telling = ["AESOP", "FlintstonesSV", "PororoSV", "VIST"]
visual_cloze = ["COMICS_Dialogue", "RecipeQA_VisualCloze"]
text_rich_vqa = ["WebQA", "TQA", "OCR-VQA", "DocVQA"]
multi_image_vqa = ["MIT-States_StateCoherence", "MIT-States_PropertyCoherence", "VISION", "RecipeQA_ImageCoherence"]

puzzle = ["RAVEN"]
nlrv2 = ["NLVR2_Mantis"]
qbench = ["QBench"]
refcoco = ["refcoco"]

def np_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

class Eval:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
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
        
    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText
    
    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip(')')
        answer = answer.strip('(')
        answer = answer.strip().lower()
        return answer

    def evaluate_rouge(self,preds):
        rouge = Rouge()
        acc = {'f': []}
        eval_list = []
        for i, res in enumerate(preds):
            sample_id = res['sample_id']
            # print(sample_id)
            gt_ans = self.process(res["gt_response"])
            pred_ans = self.process(res["pred_response"])
            # assert gt_ans != ''

            if gt_ans == '':
                continue
            
            if pred_ans == '':
                s = 0
            else:
                if len(pred_ans) > 512:
                    pred_ans = pred_ans[0: 512]
                s = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['f']
            acc['f'].append(s)
            eval_list.append({'id':str(sample_id),'score':str(round(s,3))})
        results = {'Rouge-L f': np.mean(acc['f'])}
        return results,eval_list


    def judge_multi_choice(self,sample):
        sample_id = sample['sample_id']
        gt_ans = sample["gt_response"]
        pred_ans = sample["pred_response"]

        if ":" in pred_ans:
            a_list = pred_ans.split(":")
            a_list = [a.strip() for a in a_list ]
            for a in a_list:
                if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                    pred_ans = a

        if pred_ans == gt_ans:
            return 1
        else:
            return 0

    def process_sample(self,sample):
        sample["gt_response"] = self.process(sample["gt_response"])
        sample["pred_response"] = self.process(sample["pred_response"])

    def evaluate_multichoice(self, preditions):
        correct = 0
        eval_list = []
        for i, sample in enumerate(preditions):
            self.process_sample(sample)
            score = self.judge_multi_choice(sample)
            sample_id = sample['sample_id']
            sample['result'] = score
            eval_list.append({'id':str(sample_id),'score':str(score)})
            correct+=score
        return {'Accuracy':correct/len(preditions)},eval_list
    
    def evaluate_mmvp_multichoice(self, preditions):
        correct = 0
        eval_list = []
        for i in range(0, len(preditions), 2):
            sample1, sample2 = preditions[i], preditions[i+1]
            self.process_sample(sample1)
            score1 = self.judge_multi_choice(sample1)
            sample_id1 = sample1['sample_id']
            self.process_sample(sample2)
            score2 = self.judge_multi_choice(sample2)
            sample_id2 = sample2['sample_id']
            
            score = 1 if score1 == score2 == 1 else 0
            correct+=score
            eval_list.append({"id":f"{sample_id1}_{sample_id2}","score":str(score),'id1':str(sample_id1),'score1':str(score1),'id2':str(sample_id2),'score2':str(score2)})
        
        return {'Accuracy':correct/(len(preditions)//2)},eval_list
    
    def evaluate_multi_choice_image(self,preditions):
        correct = 0
        eval_list = []
        for i,sample in enumerate(preditions):
            gt_ans = self.process(sample["gt_response"])
            pred_ans = self.process(sample["pred_response"])
            sample_id = sample['sample_id']

            if ":" in pred_ans:
                a_list = pred_ans.split(":")
                a_list = [a.strip() for a in a_list ]
                for a in a_list:
                    if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                        pred_ans = a

            if gt_ans == pred_ans:
                score = 1
            else:
                score = 0
            sample_id = sample['sample_id']
            sample['result'] = score
            eval_list.append({'id':str(sample_id),'score':str(score)})
            correct+=score
        return {'Accuracy':correct/len(preditions)},eval_list
    
    def evaluate_multi_choice_image_binary_cls(self,preditions):
        correct = 0
        eval_list = []
        auc = {"gt":[], "pre":[]}
        TP,TN,FP,FN=0,0,0,0
        for i,sample in enumerate(preditions):
            gt_ans = self.process(sample["gt_response"])
            pred_ans = self.process(sample["pred_response"])
            sample_id = sample['sample_id']

            if ":" in pred_ans:
                a_list = pred_ans.split(":")
                a_list = [a.strip() for a in a_list ]
                for a in a_list:
                    if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                        pred_ans = a
            if gt_ans.lower() == "a":
                if gt_ans.lower() == pred_ans.lower():
                    TP += 1
                else:
                    FN += 1
            else:
                if gt_ans.lower() == pred_ans.lower():
                    TN += 1
                else:
                    FP += 1

            if gt_ans == pred_ans:
                score = 1
            else:
                score = 0
                
            sample_id = sample['sample_id']
            sample['result'] = score
            eval_list.append({'id':str(sample_id),'score':str(score)})
            correct+=score
            
            if gt_ans.lower() == "a":
                auc["gt"].append(1)
            else:
                auc["gt"].append(0)
            if pred_ans.lower() == "a":
                auc["pre"].append(1)
            else:
                auc["pre"].append(0)
        precision = TP / (TP + FP) if TP != 0 else 0
        recall = TP / (TP + FN) if TP != 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if TP != 0 else 0
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(np.array(auc["gt"]), np.array(auc["pre"]))
        return {'Accuracy':correct/len(preditions), "Precision":precision, "Recall":recall, "F1":F1, "AUC": auc_score},eval_list

    def evaluate_refcoco_giou_ciou(self,preditions):
        eval_list = []
        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        for i,sample in enumerate(preditions):
            intersection += np.array(sample["intersection"])
            union += np.array(sample["union"])
            accuracy_iou += np.array(sample['accuracy_iou'])
        iou_per_class = intersection / (union + 1e-10)
        class_iou = iou_per_class[1]
        global_iou = accuracy_iou / len(preditions)
        return {"class_iou":class_iou, "global_iou":global_iou[1]}, eval_list
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, required=True)

    args = parser.parse_args()
    result_file = os.path.join(args.result_dir, "result.jsonl")
    ANOMALY_VIS = False
    
    if not os.path.exists(result_file):
        print('No prediction file found',result_file)
        exit(0)
    with open(result_file, 'r') as f:
        preds_all = [json.loads(line) for line in f]
    
    preds_all_dict = dict()
    for pred in preds_all:
        dataset_name = f"{pred['dataset']}_{pred['split_name']}"
        if dataset_name not in preds_all_dict:
            preds_all_dict[dataset_name] = list()
        preds_all_dict[dataset_name].append(pred)

    image_choice_dataset_list = ["recipeqa-RecipeQA_VisualCloze", "RecipeQA_ImageCoherence", "COMICS_Panel"]
    E = Eval()

    eval_result_list = dict()
    eval_result_list_detail = dict()

    for dataset in preds_all_dict:
        
        preds = preds_all_dict[dataset]
        question_type = preds[0]["question_type"]
        
        if question_type == "seg_single":
            eval_result, eval_list = E.evaluate_refcoco_giou_ciou(preds)
        else:
            eval_result = 'Dataset not supported'
            print('Dataset not supported')
            exit(0)

        print(dataset, end = ':  ')
        print(eval_result)

        eval_result_list[dataset] = eval_result
        eval_result_list_detail[dataset] = eval_list

    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, 'eval_dataset.json'), 'w') as f:
        json.dump(eval_result_list, f, indent=4)

    with open(os.path.join(args.result_dir,'eval_dataset_details.json'), 'w') as f:
        json.dump(eval_result_list_detail, f, indent=4)


    eval_cat_list = dict()
    print()
        
    # spot_the_diff
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in spot_the_diff:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["spot_the_diff"] = score
        print("spot_the_diff", end = ':  ')
        print('{:.2f}'.format(100 * score))

    # image_edit_instruct
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in image_edit_instruct:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["image_edit_instruct"] = score
        print("image_edit_instruct", end = ':  ')
        print('{:.2f}'.format(100 * score))

    # visual_story_telling
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in visual_story_telling:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["visual_story_telling"] = score
        print("visual_story_telling", end = ':  ')
        print('{:.2f}'.format(100 * score))

    # visual_cloze
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in visual_cloze:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["visual_cloze"] = score
        print("visual_cloze", end = ':  ')
        print('{:.2f}'.format(100 * score))

    # text_rich_vqa
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in text_rich_vqa:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["text_rich_vqa"] = score
        print("text_rich_vqa", end = ':  ')
        print('{:.2f}'.format(100 * score))

    # multi_image_vqa
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in multi_image_vqa:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["multi_image_vqa"] = score
        print("multi_image_vqa", end = ':  ')
        print('{:.2f}'.format(100 * score))

    # puzzle
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in puzzle:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["puzzle"] = score
        print("puzzle", end = ':  ')
        print('{:.2f}'.format(100 * score))

    # nlrv2
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in nlrv2:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["nlrv2"] = score
        print("nlrv2", end = ':  ')
        print('{:.2f}'.format(100 * score))

    # qbench
    score = 0
    count = 0
    for dataset in eval_result_list:
        if dataset in qbench:
            count += 1
            score += list(eval_result_list[dataset].values())[0]
    if count > 0:
        score /= count
        eval_cat_list["qbench"] = score
        print("qbench", end = ':  ')
        print('{:.2f}'.format(100 * score))

    with open(os.path.join(args.result_dir,'eval_cat.json'), 'w') as f:
        json.dump(eval_cat_list, f, indent=4)