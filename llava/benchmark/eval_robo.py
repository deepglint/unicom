import tqdm
import json
import copy
import os
import logging
import random
import numpy as np

import torch
import transformers
from dataclasses import dataclass, field
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from llava.mm_utils import get_model_name_from_path
from llava.benchmark import data_robo
from llava.model.builder import load_pretrained_model

UINFO = logging.INFO + 1

@dataclass
class EvaluationArgumentsGenerative:
    model_dir: str = field(metadata={"help": "Root directory of the data."})
    benchmarks: str = field(metadata={"help": "List of benchmarks to evaluate."})
    save_dir: str = field(default='', metadata={"help": "Output directory to save the results."})
    server_port: int = field(default=29503, metadata={"help": "Port number for the master server."})
    normalize_text: bool = field(default=True,metadata={"help": "Whether text preprocessing is used in index calculation"},)
    temperature: float = field(default=0.0, metadata={"help": "List of benchmarks to evaluate."})
    top_p: float = field(default=0.95, metadata={"help": "List of benchmarks to evaluate."})
    num_beams: int = field(default=1, metadata={"help": "List of benchmarks to evaluate."})
    max_new_tokens: int = field(default=128, metadata={"help": "max_new_tokens"})

def prepare_inputs(input, dtype, device):
    if isinstance(input, torch.Tensor):
        kwargs = {'device': device}
        if input.dtype == torch.float32 or input.dtype == torch.float16:
            kwargs['dtype'] = dtype
        input = input.to(**kwargs)
    elif isinstance(input, dict):
        for k, v in input.items():
            input[k] = prepare_inputs(v, dtype, device)
    elif isinstance(input, list):
        input = [prepare_inputs(v, dtype, device) for v in input]
    elif isinstance(input, tuple):
        input = tuple(prepare_inputs(v, dtype, device) for v in input)
    else:
        raise ValueError(f"Unsupported input type: {type(input)}")
    return input

def update_data_args(data_args, model):
    vision_tower = model.get_vision_tower()
    if "qwen" in model.config.model_type:
        data_args.def_conv_ver = "qwen_2"
    elif "llama" in model.config.model_type:
        data_args.def_conv_ver = "llava_llama_3"
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        for key in vars(data_args):
            if hasattr(model.config, key) and getattr(model.config, key, None) is not None:
                setattr(data_args, key, getattr(model.config, key, None))
    data_args.image_prefix = 'index' #default for evaluation
    print(f"data_args:{data_args}")
    return data_args

def create_model(rank, model_path):
    model_name = get_model_name_from_path(model_path)
    if 'mlcd' in model_name.lower(): # fixed loading error from "xxx/DeepGlint-AI/MLCD-Embodied-7B"
        model_name = model_name.lower().replace("mlcd", "mlcd_qwen")
    assert "llama" in model_name.lower() or "qwen" in model_name.lower(), "model_name should contain 'llama' or 'qwen'"
    model_name = model_name.lower().replace("llama", "llava_llama").replace("qwen", "llava_qwen")
    tokenizer, model, _, _ = load_pretrained_model(model_path, None, model_name, device_map=rank, torch_dtype='bfloat16')
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    return ddp_model, model, tokenizer

def run(dataloader: DataLoader, model, tokenizer, eval_args):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    model.eval()
    dataset = dataloader.dataset
    pbar_desc = f"{dataset.data_args.bmk_name}@bs{dataset.data_args.batch_size}"
    pbar = tqdm.tqdm(dataloader, desc=pbar_desc, ncols=80) if rank == 0 else dataloader
    results = []

    for _, batch in enumerate(pbar):
        batch_indices = batch.pop("indices")
        image_sizes = batch.pop("image_sizes")
        modalities = batch.pop("modalities")
        batch = prepare_inputs(batch, dtype=torch.bfloat16, device=rank)
        images = batch.pop("images")
        input_ids = batch.pop("input_ids")
        attention_mask = batch.pop("attention_mask")
        with torch.inference_mode():
            if 'llama' in model.module.config.model_type:
                output_ids = model.module.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    modalities=modalities,
                    attention_mask=attention_mask,
                    do_sample=False,
                    temperature=eval_args.temperature,
                    top_p=eval_args.top_p,
                    num_beams=eval_args.num_beams,
                    max_new_tokens=eval_args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            elif 'qwen' in model.module.config.model_type:                
                output_ids = model.module.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    modalities=modalities,
                    do_sample=False,
                    temperature=eval_args.temperature,
                    top_p=eval_args.top_p,
                    num_beams=eval_args.num_beams,
                    max_new_tokens=eval_args.max_new_tokens,
                    use_cache=True,
                )
            else:
                raise ValueError(f"Unsupported model type: {model.config.model_type}")
            
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results.extend(list(zip(batch_indices.tolist(), outputs)))

    # Build the final list of results
    fnl_results = []
    for original_id, generated_text in results:
        sample = dataset.get_sample(original_id)
        fnl_results.append({
            "unique_id": original_id, # Deduplication is done with unique id
            "question_id": sample["id"], # For easy mapping of results to the original data
            "question": sample["question"],
            "pred": generated_text.split("\n")[-1],
            "gt": sample["answer"],
            "extra_gt": sample.get("extra_answers").tolist() if sample.get("extra_answers", None) is not None else None,
            "dataset_name": dataset.data_args.bmk_name,
            "type_level_1": sample.get("type_level_1", "Undefined"),
            "type_level_2": sample.get("type_level_2", "Undefined")
            })

    if not fnl_results:
        return None

    # Synchronize data from all devices together
    dist.barrier()
    num_samples = torch.tensor(len(fnl_results), dtype=torch.int, device=rank)
    all_rank_num_samples = [torch.tensor(0, dtype=torch.int, device=rank) for _ in range(world_size)]
    dist.all_gather(all_rank_num_samples, num_samples)
    total_samples = sum(all_rank_num_samples).item()
    all_rank_results = [None] * total_samples
    dist.all_gather_object(all_rank_results, fnl_results)

    def remove_duplicates(lst):
        seen = set()
        unique_lst = []
        for item in lst:
            key = item["unique_id"]
            if key not in seen:
                seen.add(key)
                unique_lst.append(item)
        return unique_lst

    # Process all results in the main process
    if rank == 0:
        final_results = [result for result in all_rank_results if result is not None]
        final_results = [item for sublist in final_results for item in sublist]
        final_results = remove_duplicates(final_results)
        result_json_path = os.path.join(eval_args.save_dir, f"{dataset.data_args.bmk_name}.json")
        with open(result_json_path, "w", encoding="utf-8") as out_file:
            json.dump(final_results, out_file, indent=2, ensure_ascii=False)

        return final_results
    return None

def load_benchmark(rank, bmk_name, data_args, tokenizer):
    torch.cuda.set_device(rank)
    data_args = copy.deepcopy(data_args)
    data_args.bmk_name = bmk_name
    return data_robo.build_benchmark_generative(data_args, tokenizer)

def main(rank, world_size, eval_args, data_args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model, ori_model, tokenizer = create_model(rank, eval_args.model_dir)

    data_args = update_data_args(data_args, ori_model)

    for eval_bmk in eval_args.benchmarks:
        dataloader = load_benchmark(rank, eval_bmk, data_args, tokenizer)
        run(dataloader, model, tokenizer, eval_args)

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = transformers.HfArgumentParser([EvaluationArgumentsGenerative, data_robo.BenchmarkDataArguments])
    eval_args, data_args = parser.parse_args_into_dataclasses()
    if not eval_args.save_dir:
        eval_args.save_dir = os.path.join(eval_args.model_dir, 'eval_robo')
    if not os.path.exists(eval_args.save_dir):
        os.makedirs(eval_args.save_dir, exist_ok=True)

    WORLD_SIZE = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(eval_args.server_port)

    eval_args.benchmarks = eval(eval_args.benchmarks)
    logging.log(UINFO, f"Starting evaluation model {eval_args.model_dir} on following {len(eval_args.benchmarks)} benchmarks:\n{', '.join(eval_args.benchmarks)}")
    mp.spawn(main, args=(WORLD_SIZE, eval_args, data_args, ), nprocs=WORLD_SIZE, join=True)
