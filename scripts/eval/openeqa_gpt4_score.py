import json
from third_party.openeqa.evaluation.llm_match import get_llm_match_score
from collections import defaultdict
from tqdm import tqdm
import sys, os

def get_openeqa_llm_score(input_file, output_file, resume_id):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    scores_openeqa_hm3d = []
    scores_openeqa_scannet = []
    
    temp_data = []
    
    resume_flag = False

    for i, item in enumerate(tqdm(data)):
        
        if item["unique_id"] == resume_id:
            resume_flag = True
            try:
                with open(output_file, "r") as f:
                    temp_data = json.load(f)
            except FileNotFoundError:
                print("No json file to be loaded.")
        
        if resume_flag:
            question = item["question"]
            answer = item["gt"][0]
            prediction = item["pred"]
            score = get_llm_match_score(question, answer, prediction)
            llm_score = (score - 1) / 4
            item["llm_score"] = llm_score
            
            print("*" * 40)
            print("Item ID:    {}".format(i))
            print("Example question:    {}".format(question))
            print("Ground-truth answer: {}".format(answer))
            print("Predicted answer:    {}".format(prediction))
            print("LLM-match score:     {}".format(llm_score))

            temp_data.append(item)

            if item["type_level_2"] == "openeqa_hm3d-v0":
                scores_openeqa_hm3d.append(llm_score)
            elif item["type_level_2"] == "openeqa_scannet-v0":
                scores_openeqa_scannet.append(llm_score)

            if (i + 1) % 50 == 0 or (i + 1) == len(data):
                print(f"Saving temporary result into {output_file}")
                with open(output_file, 'w') as outfile:
                    json.dump(temp_data, outfile, indent=4)

    # Calculate metrics
    with open(output_file, 'r') as file:
        data = json.load(file)
    scores = defaultdict(list)

    for item in data:
        type_level_1 = item['type_level_1']
        llm_score = item['llm_score']
        scores[type_level_1].append(llm_score)
        
    for item in data:
        type_level_2 = item['type_level_2']
        llm_score = item['llm_score']
        scores[type_level_2].append(llm_score)

    for type_name, score_list in scores.items():
        average_score = sum(score_list) / len(score_list)
        print(f"Type: {type_name}, Average LLM Score: {100*average_score:.2f}")

if __name__ == "__main__":
    in_dir = sys.argv[1]
    input_file = os.path.join(in_dir, 'generative', 'openeqa.json')
    output_file = os.path.join(in_dir, 'generative', 'openeqa.json_with_llm_score.json')
    resume_id = 0 # if network is broken, we can modify the item's id to be resumed
    get_openeqa_llm_score(input_file, output_file, resume_id)
