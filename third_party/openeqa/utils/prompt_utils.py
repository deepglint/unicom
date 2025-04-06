# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

DEFAULT_DATA_DIR: Path = Path(__file__).parent.parent.parent.resolve() / "prompts"

PROMPT_NAME_TO_PATH = {
    "mmbench": DEFAULT_DATA_DIR / Path("mmbench.txt"),
    "mmbench-extra": DEFAULT_DATA_DIR / Path("mmbench-extra.txt"),
    "blind-llm": DEFAULT_DATA_DIR / Path("blind-llm.txt"),
    "gpt4v": DEFAULT_DATA_DIR / Path("gpt4v.txt"),
    "claude3-vision": DEFAULT_DATA_DIR / Path("claude3-vision.txt"),
    "gemini-pro-vision": DEFAULT_DATA_DIR / Path("gemini-pro-vision.txt"),
}


def load_prompt(name: str):
    if name not in PROMPT_NAME_TO_PATH:
        raise ValueError("invalid prompt: {}".format(name))
    path = PROMPT_NAME_TO_PATH[name]
    with path.open("r") as f:
        return f.read().strip()
