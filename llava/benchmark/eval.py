import os, tqdm, json, logging, copy
from functools import partial
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    ShardingStrategy,
    MixedPrecision,
    FullyShardedDataParallel as FSDP,
    )
from torch.distributed.fsdp.wrap import(
    transformer_auto_wrap_policy
    )

import transformers
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

import llava.benchmark.data as bmk
from llava.model import LlavaQwenForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.train.train import update_data_args

UINFO = logging.INFO + 1
logging.addLevelName(UINFO, 'UINFO')
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s]%(message)s',
                    datefmt='%y%m%d-%H%M%S', level=UINFO)

@dataclass
class EvaluationArguments:
    model_dir: str = field(metadata={"help": "Root directory of the data."})
    benchmarks: str = field(metadata={"help": "List of benchmarks to evaluate."})
    save_dir: str = field(default='', metadata={"help": "Output directory to save the results."})
    server_port: int = field(default=29503, metadata={"help": "Port number for the master server."})

def load_llava_model_for_inference(model_dir, device_map):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = LlavaQwenForCausalLM.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation='sdpa',
        torch_dtype='auto'
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def run(dataloader: DataLoader, model: torch.nn.Module):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    dataset = dataloader.dataset
    assert isinstance(dataset, bmk.BenchmarkDataset), "The dataset must be an instance of BenchmarkDataset."
    opt_ids = torch.tensor(dataset.get_opt_ids(), device=rank, dtype=torch.int)
    pbar_desc = f"{dataset.data_args.bmk_name}@bs{dataset.data_args.batch_size}"
    pbar = tqdm.tqdm(dataloader, desc=pbar_desc, ncols=80) if rank == 0 else dataloader
    results = []
    for i_batch, batch in enumerate(pbar):
        image_sizes = batch.pop("image_sizes")
        modalities = batch.pop("modalities")
        batch = bmk.prepare_inputs(batch, dtype=model.module.config.torch_dtype, device=rank)
        batch_indices = batch.pop('indices')
        with torch.no_grad():
            outputs = model.forward(**batch, return_dict=True, modalities=modalities, image_sizes=image_sizes, use_cache=False)
            for label in outputs['labels']:
                assert label.max() >= 0, f"The label maybe truncated: {label.shape} vs. {model.module.config.tokenizer_model_max_length}"
            gt = torch.eq(outputs['labels'].unsqueeze(-1), opt_ids)
            # print(f"labels:{outputs['labels']},\n opt_ids:{opt_ids},\n gt:{gt}, assert:{gt.sum(dim=[1, 2]) == 1}")
            assert torch.all(gt.sum(dim=[1, 2]) == 1)
            gt_pos = torch.argmax(gt.sum(dim=2), dim=-1) - 1
            idx_in_batch = torch.arange(gt_pos.size(0), device=gt_pos.device)

            probs = outputs['logits'].softmax(dim=-1)
            pred = probs[idx_in_batch, gt_pos][:, opt_ids].argmax(dim=-1)

        results.append(torch.stack([batch_indices, pred]).t())
    if not results:
        return None
    results = torch.cat(results)

    dist.barrier()
    num_samples = torch.scalar_tensor(results.shape[0], dtype=torch.int, device=rank)
    all_rank_num_samples = [torch.scalar_tensor(0, dtype=torch.int, device=rank) for _ in range(world_size)]
    dist.all_gather(all_rank_num_samples, num_samples)
    all_rank_results = [torch.zeros([all_rank_num_samples[i_rank], 2], dtype=torch.int64, device=rank) for i_rank in range(world_size)]
    dist.all_gather(all_rank_results, results)

    if rank == 0:
        all_rank_results = sorted(torch.cat(all_rank_results).tolist(), key=lambda x: x[0])
        opt_caps = dataset.get_all_answers()
        results = {}
        for idx, pred in all_rank_results:
            sample = dataset.get_sample(idx)
            id = sample['id']
            if id not in results:
                results[id] = {'pred': opt_caps[pred], 'gt': sample.get('answer', None)}
            elif results[id]['pred'] != opt_caps[pred]:
                if dist.get_rank() == 0:
                    logging.warning(f"The prediction results for sample {id} are inconsistent: {results[id]['pred']} vs. {pred}.")
        return results
    return None

def wrap_fsdp(model):
    trm_layer_cls = {
        Qwen2DecoderLayer,
        CLIPEncoderLayer
        }
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls = trm_layer_cls
        )
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=model.config.torch_dtype, cast_forward_inputs=True),
        device_id=dist.get_rank()
        )
    return fsdp_model

def load_benchmark(rank, bmk_name, data_args, tokenizer):
    torch.cuda.set_device(rank)
    data_args_clone = copy.deepcopy(data_args)
    data_args_clone.bmk_name = bmk_name
    return bmk.build_benchmark(data_args_clone, tokenizer)

def main(rank, world_size, eval_args: EvaluationArguments, data_args: bmk.BenchmarkDataArguments):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    model, tokenizer = load_llava_model_for_inference(eval_args.model_dir, device_map=f'cuda:{rank}')

    data_args = update_data_args(data_args, model)

    model = wrap_fsdp(model)

    for bmk_name in eval_args.benchmarks:
        dataloader = load_benchmark(rank, bmk_name, data_args, tokenizer)
        results = run(dataloader, model)
        if results is None:
            if rank == 0:
                logging.warning(f"Failed to evaluate model {eval_args.model_dir} on benchmark {bmk_name}.")
            continue
        if rank == 0:
            corrects = sum(1 for res in results.values() if res['pred'] == res['gt'])
            num_labels = sum(1 for res in results.values() if res['gt'] is not None)
            with open(os.path.join(eval_args.save_dir, f'{bmk_name}.json'), 'w') as out_file:
                json.dump(results, out_file, indent=2)
            print_msg = f"{bmk_name.upper()}: TOTAL={len(results)}, LABELS={num_labels}"
            if num_labels:
                print_msg += f", ACC={corrects/num_labels:.2%}"
            logging.log(UINFO, print_msg)
        del dataloader
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = transformers.HfArgumentParser([EvaluationArguments, bmk.BenchmarkDataArguments])
    eval_args, data_args = parser.parse_args_into_dataclasses()
    if not eval_args.save_dir:
        eval_args.save_dir = os.path.join(eval_args.model_dir, 'eval', 'multiple-choices')
    if not os.path.exists(eval_args.save_dir):
        os.makedirs(eval_args.save_dir, exist_ok=True)

    WORLD_SIZE = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(eval_args.server_port)

    eval_args.benchmarks = eval(eval_args.benchmarks)
    logging.log(UINFO, f"Starting evaluation model {eval_args.model_dir} on following {len(eval_args.benchmarks)} benchmarks:\n{', '.join(eval_args.benchmarks)}")
    mp.spawn(main, args=(WORLD_SIZE, eval_args, data_args, ), nprocs=WORLD_SIZE, join=True)
