# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base64
import os
from typing import Dict, List, Optional

import cv2
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential


def prepare_anthropic_messages(content) -> List[Dict[str, str]]:
    return [{"role": "user", "content": content}]


def prepare_anthropic_vision_messages(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    image_size: Optional[int] = 512,
):
    if image_paths is None:
        image_paths = []

    content = []
    if prefix:
        content.append({"text": prefix, "type": "text"})

    for path in image_paths:
        frame = cv2.imread(path)
        if image_size:
            factor = image_size / max(frame.shape[:2])
            frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
        _, buffer = cv2.imencode(".png", frame)
        frame = base64.b64encode(buffer).decode("utf-8")
        content.append(
            {
                "source": {"data": frame, "media_type": "image/png", "type": "base64"},
                "type": "image",
            }
        )

    if suffix:
        content.append({"text": suffix, "type": "text"})

    return [{"role": "user", "content": content}]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_anthropic_api(
    messages: List[Dict[str, str]],
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 32,
    temperature: float = 0.2,
    stop_sequences: Optional[List[str]] = None,
):
    client = Anthropic()
    message = client.messages.create(
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        stop_sequences=stop_sequences,
        temperature=temperature,
    )
    assert len(message.content) == 1
    return message.content[0].text


if __name__ == "__main__":
    assert "ANTHROPIC_API_KEY" in os.environ

    messages = prepare_anthropic_messages("Hello, Claude")
    print("input:", messages)

    # messages = prepare_anthropic_vision_messages(image_paths=["image.jpg"])

    model = "claude-3-haiku-20240307"
    output = call_anthropic_api(messages, max_tokens=32, model=model, temperature=1.0)
    print("output:", output)
