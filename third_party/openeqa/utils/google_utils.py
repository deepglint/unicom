# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import traceback
from typing import Any, List, Optional, Union

import google.generativeai as genai
from PIL.Image import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential


def set_google_key(key: Optional[str] = None) -> None:
    if key is None:
        assert "GOOGLE_API_KEY" in os.environ
        key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=key)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_google_api(
    message: Union[str, List[Union[Any, Image]]],
    model: str = "gemini-pro",  # gemini-pro, gemini-pro-vision
) -> str:
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(message)
        response.resolve()
        return response.text
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        raise e


if __name__ == "__main__":
    set_google_key(key=None)

    input = "What color are apples?"
    print("input: {}".format(input))

    output = call_google_api(input)
    print("output: {}".format(output))
