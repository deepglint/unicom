import argparse
import torch
import torch.nn.functional as F
import os

import json
from tqdm import tqdm
import shortuuid
import numpy as np

import sys
sys.path.insert(0,'.')
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_SEG_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.model import ResizeLongestSide
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava import conversation as conversation_lib

from typing import Dict, Optional, Sequence, List
import transformers
import re
import ast
import copy

from PIL import Image
import math

from pycocotools import mask as mask_util
import cv2
import tokenizers
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
IMG_SIZE = 1024
IGNORE_LABEL = 255

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape, f"output:{output.shape }, target:{target.shape}"
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    try:
        im_start, im_end = tokenizer.additional_special_tokens_ids
    except:
        im_start, im_end, _, _ = tokenizer.additional_special_tokens_ids
        
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    conv: conversation_lib.Conversation = None
) -> Dict:
    if conv is None:
        conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    sep_token_len = len(tokenizer(conv.sep).input_ids)
    sep_tail_space = int(sep[-1] == ' ')
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - sep_tail_space
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - sep_tail_space
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            if IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += sep_token_len
                instruction_len += sep_token_len
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored) \n{sources}"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "mpt" or \
       conversation_lib.default_conversation.version == "qwen_2":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def grounding_enc_processor(x: torch.Tensor) -> torch.Tensor:
    x = (x - IMG_MEAN) / IMG_STD
    h, w = x.shape[-2:]
    x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return x

def eval_model(args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.seg_token_idx = tokenizer(DEFAULT_SEG_TOKEN, add_special_tokens=False).input_ids[0]
    
    sam_transform = ResizeLongestSide(IMG_SIZE)
    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        try:
            idx = line["sample_id"]
        except:
            idx = line["id"]
        question_type = line["metadata"]["question_type"]
        dataset_name = line["metadata"]["dataset"]
        split_name = line["metadata"]["split"]

        if "image" in line:
            if isinstance(line["image"], str):
                image_files = [line["image"]]
            else:
                image_files = line["image"]
        else:
            image_files = line["images"]

        # args.conv_mode = 'mpt'
        args.conv_mode = 'qwen_2'
        
        images, image_sizes, grounding_enc_img_list, image_sam_resize_list, original_size_list = [], [], [], [], []
        for image_file in image_files:
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            images.append(image)
            image_sizes.append(image.size)
            image_np = np.array(image)
            original_size_list.append(image_np.shape[:2])
            image_np_resize = sam_transform.apply_image(image_np)
            image_sam_resize_list.append(image_np_resize.shape[:2])
            grounding_enc_img_list.append(grounding_enc_processor(torch.from_numpy(image_np_resize).permute(2, 0, 1).contiguous()).to(dtype=torch.float16, device='cuda', non_blocking=True))

        conversations = copy.deepcopy(line["conversations"])
        data_dict = preprocess([conversations], tokenizer, len(images) > 0)
        input_ids = data_dict["input_ids"][0].to(device='cuda', non_blocking=True).unsqueeze(0)
        labels = data_dict["labels"][0].to(device='cuda', non_blocking=True).unsqueeze(0)
        
        collect_size = list(set(original_size_list))
        if len(collect_size) == 0:
            mask_h, mask_w = 336, 336
        elif len(collect_size) == 1:
            mask_h, mask_w = collect_size[0]
        else:
            areas = [h*w for (h, w) in collect_size]
            mask_h, mask_w = collect_size[areas.index(max(areas))]

        masks_list = []
        for img_i in range(len(images)):
            if "segmentation" in line:
                masks = []
                for rle in line["segmentation"][img_i]:
                    m = mask_util.decode(rle)
                    m = cv2.resize(m, (mask_w, mask_h)).astype(np.uint8)
                    masks.append(m)
                masks = np.stack(masks, axis=0)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(0, mask_h, mask_w, dtype=torch.uint8)
            masks_list.append(masks)
        masks_list = torch.cat(masks_list, dim=0).float()

        image_tensors = []
        image_tensors = process_images(images, image_processor, model.config)
        if isinstance(image_tensors, list):
            image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
            if image_aspect_ratio=="anyres_mul" or image_aspect_ratio=="anyres":
                image_tensors = [[x_.to(dtype=torch.float16, device='cuda', non_blocking=True)for x_ in image_tensors]]
            else:
                image_tensors = [[x_.unsqueeze(dim=0).to(dtype=torch.float16, device='cuda', non_blocking=True) for x_ in image_tensors]]
        else:
            image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():
            output = model.forward(
                input_ids=input_ids,
                attention_mask=None,
                inputs_embeds = None,
                labels=labels,
                output_attentions=False,
                output_hidden_states=True,
                images=image_tensors,
                image_sizes = image_sizes,
                grounding_enc_imgs = [torch.stack(grounding_enc_img_list, dim=0)],
                image_sam_resizes = [image_sam_resize_list],
                original_sizes = [(mask_h, mask_w)],
                infer=True,
                )

        pred_mask = output["pred_masks"][0]
            
        pred_mask = (pred_mask > 0).int()  # Thresholding to get binary masks
        masks_list = (masks_list > 0).int()

        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        for target, prediction in zip(masks_list, pred_mask):
            target = target.to(prediction.device)
            intersect, union_, _ = intersectionAndUnionGPU(
                prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
            )
            intersection += intersect
            union += union_
            accuracy_iou += intersect / (union_ + 1e-5)
            # handles no-object targets
            accuracy_iou[union_ == 0] += 1.0
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        accuracy_iou = accuracy_iou.cpu().numpy() / masks_list.shape[0]
        
        intersection = [float(intersection[0]), float(intersection[1])]
        union = [float(union[0]), float(union[1])]
        accuracy_iou = [float(accuracy_iou[0]), float(accuracy_iou[1])]

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "dataset": dataset_name,
                                   "split_name":split_name,
                                   "sample_id": idx,
                                   "intersection":intersection,
                                   "accuracy_iou":accuracy_iou,
                                   "union":union,
                                   "shortuuid": ans_id,
                                   "model_id": model_name,
                                   "question_type": question_type,
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)