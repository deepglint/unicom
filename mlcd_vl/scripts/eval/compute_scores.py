import sys, os, json
import glob
from collections import defaultdict
from scripts.eval.robo_metric import (
    get_robovqa_score,
    get_openeqa_score
    )

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

def compute_scores(in_dir):
    result_json_path = glob.glob(os.path.join(in_dir, "*.json"))
    metric_info_all = {}

    for bmk_path in result_json_path:
        bmk_name = os.path.basename(bmk_path).split(".")[0]
        if "robovqa" in bmk_path:
            score_info, contents = get_robovqa_score(bmk_path)
            metric_info_all[f"{bmk_name}"] = (len(contents), score_info["score"] * 100)
            metric_info_all[f"{bmk_name}_bleu1"] = (len(contents), score_info["bleu1"] * 100)
            metric_info_all[f"{bmk_name}_bleu2"] = (len(contents), score_info["bleu2"] * 100)
            metric_info_all[f"{bmk_name}_bleu3"] = (len(contents), score_info["bleu3"] * 100)
            metric_info_all[f"{bmk_name}_bleu4"] = (len(contents), score_info["bleu4"] * 100)
        elif "openeqa" in bmk_path:
            score_info, contents = get_openeqa_score(bmk_path)
            for type_name, score_item in score_info.item():
                metric_info_all[f"{bmk_name}_{type_name}"] = (len(contents), score_item * 100)

    print_scores('GENERATIVE', metric_info_all)

if __name__ == '__main__':
    in_dir = sys.argv[1]
    compute_scores(in_dir)
