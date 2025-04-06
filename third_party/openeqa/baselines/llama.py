# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path
from typing import Optional

import tqdm

from third_party.openeqa.utils.llama_utils import LLaMARunner, enable_full_determinism
from third_party.openeqa.utils.prompt_utils import load_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to EQA dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        required=True,
        help="path to weights in huggingface format",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="model name (defaults to model path folder name)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="load model in 8bit mode (default: false)",
    )
    parser.add_argument(
        "--use-fast-kernels",
        action="store_true",
        help="use fast kernels (default: false)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="gpt seed (default: 1234)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="gpt temperature (default: 0.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="gpt maximum tokens (default: 128)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    enable_full_determinism(args.seed)
    if args.model_name is None:
        args.model_name = args.model_path.name.lower()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}-{}.json".format(args.model_name, args.seed)
    )
    return args


def parse_output(output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()


def ask_question(
    model, question: str, max_tokens: int = 128, temperature: float = 0.2
) -> Optional[str]:
    prompt = load_prompt("blind-llm")
    input = prompt.format(question=question)
    output = model(input, max_new_tokens=max_tokens, temperature=temperature)
    return parse_output(output)


def main(args: argparse.Namespace):
    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load model
    model = LLaMARunner(
        args.model_path,
        load_in_8bit=args.load_in_8bit,
        use_fast_kernels=args.use_fast_kernels,
    )

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        # generate answer
        question = item["question"]
        answer = ask_question(model=model, question=question)

        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())
