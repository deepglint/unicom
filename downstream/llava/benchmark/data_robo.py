import os, logging, pickle, re
import pandas as pd
from typing import Dict
from pathlib import Path

from dataclasses import dataclass, field
from typing import Dict, Sequence
from transformers import PreTrainedTokenizer as PTTokenizer
import torch, torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from llava.train import train
from llava import conversation as conversation_lib

@dataclass
class BenchmarkDataArguments(train.DataArguments):
    bmk_name: str = field(default='', metadata={"help": "Name of the benchmark."})
    bmk_root: str = field(default='/path/to/benchmarks', metadata={"help": "Root directory of the data."})
    batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation."})
    max_samples: int = field(default=0, metadata={"help": "Maximum number of evaluation steps."})
    def_conv_ver: str = field(default='qwen_2', metadata={"help": "Version of default conversation template"})
    num_workers: int = field(default=0, metadata={"help": "Number of workers for data loading."})

@dataclass
class DataCollatorForBenchmarkDataset(train.DataCollatorForSupervisedDataset):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        indices = [instance['index'] for instance in instances]
        batch['indices'] = torch.tensor(indices, dtype=torch.int)
        return batch

def load_bmk_data(dir, format='parquet', prefix='', reset_index=True, **kwargs):
    assert format == 'parquet' or format == 'csv', f"Unsupported format: {format}"
    def _load(dir, prefix, reset_index):
        file_list = list(Path(dir).rglob(f'{prefix}*.{format}'))
        assert len(file_list) != 0, f"No parquet files found in {dir}"
        file_list = sorted([str(file_path) for file_path in file_list])
        if format == 'parquet':
            df_list = [pd.read_parquet(file_path, **kwargs) for file_path in file_list]
        elif format == 'csv':
            df_list = [pd.read_csv(file_path, **kwargs) for file_path in file_list]
        if len(df_list) == 1:
            df = df_list[0]
        else:
            df = pd.concat(df_list, ignore_index=True)
        if reset_index:
            df.reset_index(drop=True, inplace=True)
        return df

    if dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            df = _load(dir, prefix, reset_index)
            buffer_list = [pickle.dumps(df)]
        else:
            buffer_list = [None]
        dist.broadcast_object_list(buffer_list, src=0)
        df = pickle.loads(buffer_list[0])
    else:
        df = _load(dir, prefix, reset_index)
    return df


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

class BenchmarkDatasetGenerative(train.LazySupervisedDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        Dataset.__init__(self)
        self.max_samples = data_args.max_samples
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.df = []
        conversation_lib.default_conversation = conversation_lib.conv_templates[self.data_args.def_conv_ver]
        self.default_answer = None

    def __len__(self):
        return min(len(self.df), self.max_samples) if self.max_samples > 0 else len(self.df)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self):
            raise IndexError(f"Index {i} out of range for dataset of length {len(self)}")

        sample = self.get_sample(i)
        source = self._sample2source(sample)

        source['conversations'][0]['value'] = self._remove_image_tags(
                source['conversations'][0]['value'], len(source['images']))
        assert '<image>' not in source['conversations'][0]['value'], \
            f"Found <image> in source:\n{source['conversations'][0]['value']}"
        
        if self.data_args.image_prefix == 'naked':
                img_tag = "<image>" * len(source['images'])
        elif self.data_args.image_prefix == 'naked_line':
            img_tag = "<image>" * len(source['images']) + "\n"
        elif self.data_args.image_prefix == 'naked_lines':
            img_tag = "<image>\n" * len(source['images'])
        elif self.data_args.image_prefix == 'patch':
            img_tag = "<image>\n" * len(source['images'])
        elif self.data_args.image_prefix.startswith("index"):
            img_tag = ""
            for i_img in range(len(source['images'])):
                img_tag += f"Image {i_img + 1}:\n"
                img_tag += "<image>\n"
        else:
            raise ValueError(f"Unsupported image prefix mode: {self.data_args.image_prefix}")
        
        source['conversations'][0]['value'] = img_tag + source['conversations'][0]['value']

        num_img_tags = 0
        for i_msg, msg in enumerate(source['conversations']):
            num_img_tags += msg['value'].count('<image>')
        assert num_img_tags == len(source['images']), f"{source['id']} {msg['value']}"
        
        item = self._source2item(source)
        item = self._update_input_ids(item)

        item['index'] = i
        return item

    def _remove_image_tags(self, question: str, num_imgs) -> str:
        if num_imgs:
            num_tags = question.count('<image>')
            assert num_tags == 0 or num_tags == num_imgs, question
        else:
            assert question.count('<image>') == 0, question
        if not hasattr(self, 'img_tag_pattern'):
            self.img_tag_pattern = re.compile(r'^(?:(?:[Ii]mage [1-9][0-9]?:\s)?<image>\s?)+')
        return self.img_tag_pattern.sub('', question)

    def get_sample(self, i: int) -> Dict:
        err_msg = "A subclass of BenchmarkDataset must implement the get_sample method."
        raise NotImplementedError(err_msg)

    def _sample2source(self, sample) -> Dict:
        placeholders = {
            'QUESTION': sample['question'].strip()
        }
        if 'images' in sample and len(sample['images']) > 0:
            if len(sample['images']) == 1:
                default_template = "Please observe the image carefully and answer following question based on the image.\n{QUESTION}\n"
            else:
                default_template = "Please observe these " + str(len(sample['images'])) + \
                    " images carefully and answer following question based on them.\n{QUESTION}\n"
        else:
            default_template = "{QUESTION}\n"
        
        default_template += "Your answer is:"

        template = sample.get('template', default_template)
        question = template.format(**placeholders)

        answer = sample.get('answer', self.default_answer)
        assert answer, f"Answer is not provided for sample {sample['id']} and no default answer is set."
        logging.info(question)

        source = {
            'id': sample['id'],
            'conversations': [
                {'from': 'human', 'value': question},
                {'from': 'gpt', 'value': answer}
                ]
            }
        if 'images' in sample:
            source['images'] = sample['images']
        return source
    
    def _update_input_ids(self, item: Dict) -> Dict[str, torch.Tensor]:
        question_length = ((item["labels"] >= 0).int() == 0).nonzero(as_tuple=True)[0][-1] + 3
        item.update({"input_ids": item["input_ids"][:question_length]})
        return item

@register_generative_benchmark('robovqa')
class RoboVQA_Dataset(BenchmarkDatasetGenerative):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'RoboVQA'))
        self.prefix_yes_or_no = " Please answer yes or no. "
    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        if ". is it" in row['question']:
          sample = {
              'id': str(row['id']),
              'question': row['question'] + self.prefix_yes_or_no,
              'images': [row[f'image{id+1}'] for id in range(8)],
              'answer': row['answer']
          }
        else:           
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
            'extra_answers': row['extra_answers'],
            "type_level_1": row["type"].split('_')[-1],
            "type_level_2": '_'.join(row["type"].split('_')[:2])
        }
        return sample