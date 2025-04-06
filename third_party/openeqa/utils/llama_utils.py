# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Adapted from: https://github.com/meta-llama/llama-recipes/blob/df03fd4b1247401218c11bbc0e2a46441bc955c5/recipes/inference/local_inference/inference.py
"""

import argparse
from pathlib import Path
from typing import List, Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def enable_full_determinism(seed: int):
    transformers.enable_full_determinism(seed)


class LLaMARunner:
    def __init__(
        self,
        model: Union[str, Path],
        load_in_8bit: bool = False,
        use_fast_kernels: bool = False,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            return_dict=True,
            load_in_8bit=load_in_8bit,
            device_map="auto",
            attn_implementation="sdpa" if use_fast_kernels else None,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __call__(
        self,
        input: str,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        top_p: float = 1.0,
        top_k: int = 50,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: int = 1,
        use_cache: bool = True,
    ) -> str:
        batch = self.tokenizer(input, padding=True, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                use_cache=use_cache,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            )
        batch_length = batch["input_ids"].shape[1]
        output_text = self.tokenizer.decode(
            outputs[0][batch_length:], skip_special_tokens=True
        )
        return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="path to llama-2 weights (in huggingface format)",
        required=True,
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
    args = parser.parse_args()
    llama = LLaMARunner(
        args.model,
        load_in_8bit=args.load_in_8bit,
        use_fast_kernels=args.use_fast_kernels,
    )
    input = "I have tomatoes, basil and cheese at home. What can I cook for dinner?\n"
    output = llama(input, max_new_tokens=512, do_sample=False)
    print("Q: {}".format(input.strip()))
    print("A: {}".format(output.strip()))
