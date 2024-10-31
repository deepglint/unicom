# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import traceback, os
from typing import Optional

from third_party.openeqa.utils.openai_utils import (
    call_openai_api,
    prepare_openai_messages,
    set_openai_key,
)
from third_party.openeqa.utils.prompt_utils import load_prompt
import requests

def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        return "Invalid output string: {}".format(output)
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())

def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    openai_model: str = "gpt-4-1106-preview",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    openai_top_p: float = 0.95,
    verbose: bool = False,
    endpoint: bool = False,
):
    if prediction is None:
        return 0

    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = load_prompt(prompt_name)

    if endpoint:
        API_KEY = os.environ["YOUR_API_KEY"]
        ENDPOINT = os.environ["YOUR_ENDPOINT"]
        # use requesets to access API
        try:
            headers = {
                "Content-Type": "application/json",
                "api-key": API_KEY,
            }

            messages = prepare_openai_messages(
                prompt.format(
                    question=question,
                    answer=answer,
                    prediction=prediction,
                    extra_answers=extra_answers,
                ),
            )
            payload = {
                "messages": messages,
                "temperature": openai_temperature,
                "top_p": openai_top_p,
                "max_tokens": openai_max_tokens,
            }

            response = requests.post(ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad responses
            response = response.json()
            content = response['choices'][0]['message']['content']
            return parse_score(content)

        except Exception as e:
            traceback.print_exc()
            raise e

    else:
        API_KEY = os.environ["YOUR_API_KEY"]
        # use defualt method to access API
        try:
            set_openai_key(key=API_KEY)
            messages = prepare_openai_messages(
                prompt.format(
                    question=question,
                    answer=answer,
                    prediction=prediction,
                    extra_answers=extra_answers,
                ),
            )
            output = call_openai_api(
                messages=messages,
                model=openai_model,
                seed=openai_seed,
                max_tokens=openai_max_tokens,
                temperature=openai_temperature,
                verbose=verbose,
            )
            return parse_score(output)
        except Exception as e:
            traceback.print_exc()
            raise e


if __name__ == "__main__":
    # example usage
    question = "What color is the rug?"
    answer = "tan with pink and blue"
    prediction = "brown with pink and blue"
    score = get_llm_match_score(question, answer, prediction, endpoint=True)
    print("*" * 40)
    print("example question:    {}".format(question))
    print("ground-truth answer: {}".format(answer))
    print("predicted answer:    {}".format(prediction))
    print("llm-match score:     {}".format(score))
    print("*" * 40)
