import os
from typing import Dict

from transformers import PreTrainedTokenizer as PTTokenizer
import torch, torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from llava.benchmark.data import (
    BenchmarkDataset,
    BenchmarkDataArguments,
    load_bmk_data,
    DataCollatorForBenchmarkDataset
)

generative_benchmark_classes = {}

def get_generative_benchmark_by_name(name):
    assert name in generative_benchmark_classes, f"Benchmark name '{name}' is not registered."
    return generative_benchmark_classes.get(name)

def register_generative_benchmark(name):
    def decorator(cls):
        generative_benchmark_classes[name] = cls
        return cls
    return decorator

def build_benchmark_generative(data_args: BenchmarkDataArguments, tokenizer: PTTokenizer) -> Dict:
    name_items = data_args.bmk_name.split(".")
    kwargs = {"data_args": data_args, "tokenizer": tokenizer}
    if len(name_items) > 1:
        kwargs["subset"] = ".".join(name_items[1:])
    dataset = get_generative_benchmark_by_name(name_items[0])(**kwargs)
    assert len(dataset.df) > 0, f"No available data in benchmark {data_args.bmk_name}."

    sampler = None
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=0,
            drop_last=False
        )
    mp.set_start_method('fork', force=True)
    dataloader = DataLoader(
        dataset,
        shuffle=(sampler is None),
        batch_size=data_args.batch_size,
        sampler=sampler,
        num_workers=data_args.num_workers,
        collate_fn=DataCollatorForBenchmarkDataset(tokenizer=tokenizer)
    )
    return dataloader

class BenchmarkDatasetGenerative(BenchmarkDataset):
    def _source2item(self, source: Dict) -> Dict[str, torch.Tensor]:
        item = super()._source2item(source)
        # item = self._update_input_ids(item)
        return item

    def _update_input_ids(self, item: Dict) -> Dict[str, torch.Tensor]:
        question_length = (item["labels"] >= 0).int().argmax()
        item.update({"input_ids": item["input_ids"][:question_length]})
        return item

@register_generative_benchmark('ocrbench')
class OcrBench_Dataset(BenchmarkDatasetGenerative):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, "ocrbench", "data"))

    def get_sample(self, i) -> Dict:
        row = self.df.iloc[i]
        sample = {
            "id": str(i),
            "answer": "[SEG]".join(list(row["answer"])),
            "question": row["question"],
            "images": [row["image"]["bytes"]],
            "type_level_1": row["question_type"],
            "type_level_2": row["dataset"]
        }
        return sample

@register_generative_benchmark('textvqa')
class TextVQA_Dataset(BenchmarkDatasetGenerative):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, "textvqa", "data", "val"))
        # In order to compare it with lmms indicators
        self.prompt_suffix = "\nAnswer the question using a single word or phrase."

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            "id": str(row["question_id"]),
            "answer": "[SEG]".join(list(row["answers"])),
            "question": row["question"] + self.prompt_suffix,
            "images": [row["image"]["bytes"]]
        }
        return sample

@register_generative_benchmark('chartqa')
class ChartQA_Dataset(BenchmarkDatasetGenerative):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, "chartqa", "data"))
        # In order to compare it with lmms indicators
        self.prompt_suffix = "\nAnswer the question with a single word."

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            "id": str(i),
            "answer": row["answer"],
            "question": row["question"] + self.prompt_suffix,
            "images": [row["image"]["bytes"]]
        }
        return sample

@register_generative_benchmark('docvqa')
class DocVQA_Dataset(BenchmarkDatasetGenerative):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, "docvqa", "data", "val"))
        # In order to compare it with lmms indicators
        self.prompt_suffix = "\nAnswer the question using a single word or phrase."

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            "id": str(row["questionId"]),
            "answer": "[SEG]".join(list(row["answers"])),
            "question": row["question"] + self.prompt_suffix,
            "images": [row["image"]["bytes"]]
        }
        return sample

@register_generative_benchmark('robovqa')
class RoboVQA_Dataset(BenchmarkDatasetGenerative):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'RoboVQA'))
    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            'id': str(row['id']),
            'question': row['question'],
            'images': [row[f'image{id+1}'] for id in range(8)],
            'answer': row['answer']
        }
        return sample
    
@register_generative_benchmark('openeqa')
class OpenEQA_Dataset(BenchmarkDatasetGenerative):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'OpenEQA'))
    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            'id': str(row['id']),
            'question': row['question'],
            'images': row['images'].tolist(),
            'answer': row['answer'],
            'extra_answer': row['extra_answer'],
            "type_level_1": row["type"].split('_')[-1],
            "type_level_2": '_'.join(row["type"].split('_')[:2])
        }
        return sample