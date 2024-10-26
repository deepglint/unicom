import sys, os, json
import glob
from collections import defaultdict
from llava.benchmark.data import load_bmk_data
from scripts.eval.generative_metric import (
    get_chartqa_score,
    reformat_chartqa_result,
    get_docvqa_score,
    reformat_docvqa_result,
    get_textvqa_score,
    reformat_textvqa_result,
    get_ocrbench_score,
    reformat_ocrbench_result,
    get_robovqa_score,
    reformat_robovqa_result
    )

bmk_root = '/home/vlm/benchmarks'

def mme(data):
    results = []
    df = load_bmk_data(os.path.join(bmk_root, 'MME', 'data'))
    for i, row in df.iterrows():
        id = str(i)
        assert id in data
        item = data[id]

        results.append({
            'question_id': row['question_id'],
            'score': 1.0 if item['gt'] == item['pred'] else 0.0,
            'category': row['category']
        })

    category2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        category = result["category"]
        if question_id not in category2score[category]:
            category2score[category][question_id] = []
        category2score[category][question_id].append(score)
    category2avg_score = {}
    for category, question2scores in category2score.items():
        total_score = 0
        for question_id, scores in question2scores.items():
            assert len(scores) == 2, f"question_id={question_id}, scores={scores}"
            acc = sum(scores) / len(scores) * 100.0
            acc_plus = (sum(scores) == 2) * 100.0
            score = acc_plus + acc
            total_score += score
        avg_score = total_score / len(question2scores)
        category2avg_score[category] = avg_score
    total_score = sum(category2avg_score.values())
    return f'{total_score:.0f}'

def acc(data):
    num_labels = 0
    corrects = 0
    for val in data.values():
        assert isinstance(val, dict) and 'pred' in val, f"{val}"
        pred = val['pred']
        gt = val.get('gt', None)
        if gt is not None:
            corrects += (pred == gt)
            num_labels += 1
    return ((corrects / num_labels) * 100) if num_labels > 0 else 'N/A'

def merge_keys(results: dict, keys: list):
    num_total, sum_score = 0, 0
    if all(key in results for key in keys):
        common_prefix = os.path.commonprefix(keys).rstrip('.')
        for key in keys:
            assert key in results
            num, score = results[key]
            num_total += num
            sum_score += score * num
            results.pop(key)
        results[common_prefix] = (num_total, sum_score / num_total)
    return results

def print_scores(title, named_num_scores): # dict[name: str]: (num: int, score: float or str)
    if len(named_num_scores) > 0:
        max_name_len = max([len(name) for name in named_num_scores.keys()])
        name_width = max(max_name_len, len(title)) + 2
        print(f"\n{title:<{name_width}} {'NUM.':<10}{'SCORE':<10}")
        for name, (num, score) in named_num_scores.items():
            if isinstance(score, float):
                score = f'{score:.2f}'
            else:
                assert isinstance(score, str)
            print(f"{name:<{name_width}}{num:<10}{score:<10}")

def compute_multiple_choices(in_dir):
    print(f"Computing accuracy for files in {in_dir}")
    results = {}
    for file_name in os.listdir(in_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(in_dir, file_name)
            with open(file_path, 'r') as in_file:
                data = json.load(in_file)
                if len(data) > 0:
                    bmk_name = os.path.splitext(file_name)[0]
                    if bmk_name == 'mme':
                        score = mme(data)
                    elif bmk_name.startswith('mmbench'):
                        if 'test' in bmk_name:
                            score = 'N/A'
                        else:
                            score = acc(data)
                    elif bmk_name.startswith('mmmu'):
                        if 'test' in bmk_name:
                            score = 'N/A'
                        else:
                            score = acc(data)
                    else:
                        score = acc(data)
                    results[bmk_name] = (len(data), score)

    results = merge_keys(results, ['scienceqa.test', 'scienceqa.validation'])
    results = merge_keys(results, ['mmmu.dev', 'mmmu.validation'])
    results = merge_keys(results, ['cmmmu.dev', 'cmmmu.val'])
    results = merge_keys(results, ['mmlu.dev', 'mmlu.test', 'mmlu.val'])
    results = merge_keys(results, ['cmmlu.dev', 'cmmlu.test'])
    print_scores('MULTI-CHOICES', results)

def compute_generative(in_dir):
    result_json_path = glob.glob(os.path.join(in_dir, "*.json"))
    metric_info_all = {}

    for bmk_path in result_json_path:
        bmk_name = os.path.basename(bmk_path).split(".")[0]
        if "ocrbench" in bmk_path:
            contents = reformat_ocrbench_result(bmk_path)
            correct_count = get_ocrbench_score(contents)
            metric_info_all[f"{bmk_name}"] = (len(contents), correct_count * 100)
        elif "chartqa" in bmk_path:
            contents = reformat_chartqa_result(bmk_path)
            relaxed_overall = get_chartqa_score(contents)
            metric_info_all[f"{bmk_name}"] = (len(contents), relaxed_overall * 100)
        elif "docvqa" in bmk_path:
            contents = reformat_docvqa_result(bmk_path)
            anls_socre = get_docvqa_score(contents)
            metric_info_all[f"{bmk_name}"] = (len(contents), anls_socre * 100)
        elif "textvqa" in bmk_path:
            contents = reformat_textvqa_result(bmk_path)
            exact_match = get_textvqa_score(contents)
            metric_info_all[f"{bmk_name}"] = (len(contents), exact_match * 100)
        elif "robovqa" in bmk_path:
            contents = reformat_robovqa_result(bmk_path)
            score_info = get_robovqa_score(contents)
            metric_info_all[f"{bmk_name}"] = (len(contents), score_info["score"] * 100)
            metric_info_all[f"{bmk_name}_bleu1"] = (len(contents), score_info["bleu1"] * 100)
            metric_info_all[f"{bmk_name}_bleu2"] = (len(contents), score_info["bleu2"] * 100)
            metric_info_all[f"{bmk_name}_bleu3"] = (len(contents), score_info["bleu3"] * 100)
            metric_info_all[f"{bmk_name}_bleu4"] = (len(contents), score_info["bleu4"] * 100)
        elif "openeqa" in bmk_path:
            print("Please use GPT-4-1103-Preview model to evaluate the score for OpenEQA.")

    print_scores('GENERATIVE', metric_info_all)

if __name__ == '__main__':
    in_dir = sys.argv[1]
    compute_multiple_choices(os.path.join(in_dir, 'multiple-choices'))
    compute_generative(os.path.join(in_dir, 'generative'))
