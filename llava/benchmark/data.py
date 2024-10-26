import os, logging, string, re, pickle, pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Sequence
from pathlib import Path
from functools import partial

import torch, torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import PreTrainedTokenizer as PTTokenizer

from llava.train import train
from llava import conversation as conversation_lib

@dataclass
class BenchmarkDataArguments(train.DataArguments):
    bmk_name: str = field(default='', metadata={"help": "Name of the benchmark."})
    bmk_root: str = field(default='/home/vlm/benchmarks', metadata={"help": "Root directory of the data."})
    batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation."})
    max_samples: int = field(default=0, metadata={"help": "Maximum number of evaluation steps."})
    def_conv_ver: str = field(default='qwen_2', metadata={"help": "Version of default conversation template"})
    max_num_images: int = field(default=1, metadata={"help": "Maximum number of images per sample."})
    num_workers: int = field(default=0, metadata={"help": "Number of workers for data loading."})

@dataclass
class DataCollatorForBenchmarkDataset(train.DataCollatorForSupervisedDataset):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        indices = [instance['index'] for instance in instances]
        batch['indices'] = torch.tensor(indices, dtype=torch.int)
        return batch

benchmark_classes = {}
def register_benchmark(name):
    def decorator(cls):
        benchmark_classes[name] = cls
        return cls
    return decorator

def get_benchmark_by_name(name):
    assert name in benchmark_classes, f"Benchmark name '{name}' is not registered."
    return benchmark_classes.get(name)

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

class BenchmarkDataset(train.LazySupervisedDataset):
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
                source['conversations'][0]['value'], len(source['image']))
        assert '<image>' not in source['conversations'][0]['value'], \
            f"Found <image> in source:\n{source['conversations'][0]['value']}"
        
        if self.data_args.image_prefix == 'naked':
                img_tag = "<image>" * len(source['image'])
        elif self.data_args.image_prefix == 'naked_line':
            img_tag = "<image>" * len(source['image']) + "\n"
        elif self.data_args.image_prefix == 'naked_lines':
            img_tag = "<image>\n" * len(source['image'])
        elif self.data_args.image_prefix == 'patch':
            img_tag = "<image>\n" * len(source['image'])
        elif self.data_args.image_prefix.startswith("index"):
            img_tag = ""
            for i_img in range(len(source['image'])):
                img_tag += f"Image {i_img + 1}:\n"
                img_tag += "<image>\n"
        else:
            raise ValueError(f"Unsupported image prefix mode: {self.data_args.image_prefix}")
        
        source['conversations'][0]['value'] = img_tag + source['conversations'][0]['value']

        num_img_tags = 0
        for i_msg, msg in enumerate(source['conversations']):
            num_img_tags += msg['value'].count('<image>')
        assert num_img_tags == len(source['image']), f"{source['id']} {msg['value']}"
        
        item = self._source2item(source)

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

    def get_opt_ids(self):
        opt_ids = self.tokenizer(self.get_all_answers(), return_attention_mask=False)['input_ids']
        assert all(len(id) == 1 for id in opt_ids)
        return [id[0] for id in opt_ids]

    def get_all_answers(self) -> list:
        raise NotImplementedError(
            "A subclass of BenchmarkDataset must implement the get_all_answers method.\n" \
            "Returns the all available answers in the dataset."
            )

    def get_sample(self, i: int) -> Dict:
        err_msg = """\
A subclass of BenchmarkDataset must implement the get_sample method.
This method shoud return a dictionary with the following format:
{
    'id': str,
    'image':  bytes of image file,
    'question': str,
    'options': List[str], without the caps like A., B., C., D.,
    'answer': str like 'A', 'B', 'C', 'D',
    'template': 'prefix\n{QUESTION}\n{OPTIONS}\nsuf{OPT_CAPS}fix"
}"""
        raise NotImplementedError(err_msg)

    def _format_opt_caps(self, sample, lang='en') -> str:
        opt_caps = self.get_all_answers()[:len(sample['options'])]
        if lang == 'en':
            if len(opt_caps) > 2:
                opt_caps_str = ', '.join(opt_caps[:-1]) + ', or ' + opt_caps[-1]
            else:
                opt_caps_str = ' or '.join(opt_caps)
        elif lang == 'zh':
            opt_caps = self.get_all_answers()[:len(sample['options'])]
            if len(opt_caps) > 2:
                return '，'.join(opt_caps[:-1]) + '或' + opt_caps[-1]
            else:
                return '或'.join(opt_caps)
        else:
            raise ValueError(f"Unsupported language: {lang}")
        return opt_caps_str

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
        if 'options' in sample:
            opt_fmt = lambda j: f"{self.get_all_answers()[j]}. {sample['options'][j]}"
            placeholders['OPTIONS'] = '\n'.join([opt_fmt(j) for j in range(len(sample['options']))])
            placeholders['OPT_CAPS'] = self._format_opt_caps(sample)
            default_template += "{OPTIONS}\n\nPlease select the correct answer by writing the letter ({OPT_CAPS}) that precedes your choice.\n"
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
            source['image'] = sample['images']
        return source

def build_benchmark(data_args: BenchmarkDataArguments, tokenizer: PTTokenizer) -> Dict:
    name_items = data_args.bmk_name.split('.')
    kwargs = {'data_args': data_args, 'tokenizer': tokenizer}
    if len(name_items) > 1:
        kwargs['subset'] = '.'.join(name_items[1:])
    dataset = get_benchmark_by_name(name_items[0])(**kwargs)
    assert len(dataset.df) > 0, f"No available data in benchmark {data_args.bmk_name}"

    sampler = None
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset,
            num_replicas = world_size,
            rank = rank,
            shuffle = False,
            seed = 0,
            drop_last = False
            )
    mp.set_start_method('fork', force=True)
    dataloader = DataLoader(
        dataset,
        shuffle = (sampler is None),
        batch_size = data_args.batch_size,
        sampler = sampler,
        num_workers = data_args.num_workers,
        collate_fn=DataCollatorForBenchmarkDataset(tokenizer=tokenizer)
        )
    return dataloader

@register_benchmark('ai2d')
class AI2D_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'ai2d', 'data'))
        def is_opt_idx(ans):
            return isinstance(ans, str) and ans.isnumeric() and 0 <= int(ans) <= 3
        self.df = self.df.loc[self.df['answer'].apply(is_opt_idx)]
        self.df['answer'] = self.df['answer'].astype(int)

    def get_all_answers(self) -> list:
        options = sorted(list(set(self.df['answer'])))
        assert len(options) == 4
        return list(string.ascii_uppercase[:len(options)])

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        opt_caps = 'ABCD'
        question = row['question'].strip().capitalize()
        question = question.replace('  ', ' ').replace(' ?', '?')
        if bool(re.search(r'[a-zA-Z]$', question)):
            if question.split(' ')[0] in ['How', 'What', 'Which', 'Where', 'When', 'Who', 'Why'] and question[-1] != '?':
                question += '?'
        return {
            'id': str(i),
            'images': [row['image']['bytes']],
            'question': question,
            'options': row['options'],
            'answer': opt_caps[row['answer']]
        }

@register_benchmark('mme')
class MME_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'MME', 'data'))
        self.with_options = False

    def get_all_answers(self) -> list:
        options = sorted(list(set(self.df['answer'])))
        assert len(options) == 2
        return options

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        return {
            'id': str(i),
            'images': [row['image']['bytes']],
            'question': row['question'],
            'answer': row['answer']
        }

@register_benchmark('mmbench')
class MMBench_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer, subset):
        super().__init__(data_args, tokenizer)
        assert '.' in subset
        self.lang, self.split = subset.split('.')
        assert self.split == 'dev' or self.split == 'test', f"Unsupported subset: {subset}"
        if not any([self.lang != lang for lang in ['en', 'cn', 'cc']]):
            raise ValueError(f"Unsupported language: {self.lang}")
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'MMBench', self.lang), prefix=self.split)
        if self.split == 'test':
            self.df.drop(columns=['answer'], inplace=True)
            self.default_answer = 'A'

    def get_all_answers(self) -> list:
        return ['A', 'B', 'C', 'D']

    def _format_opt_caps(self, sample) -> str:
        lang = self.lang if self.lang == 'en' else 'zh'
        return super()._format_opt_caps(sample, lang)

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        options = list(filter(lambda opt: opt != 'nan', [row['A'], row['B'], row['C'], row['D']]))
        sample = {
            'id': str(row['index']),
            'images': [row['image']['bytes']],
            'question': row['question'],
            'options': options,
        }
        if 'answer' in row:
            sample['answer'] = row['answer']
        if self.lang == 'cn' or self.lang == 'cc':
            sample['template'] = "请仔细观察这幅图像，并根据图像的内容回答下面的问题。\n{QUESTION}\n{OPTIONS}\n\n" \
                "请选择正确答案并写出答案前的标号 ({OPT_CAPS}）。\n你的答案是："
        return sample

@register_benchmark('nlvr2')
class NLVR2_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        assert data_args.max_num_images < 0 or data_args.max_num_images >= 2, "Each sample in NLVR2 has two images."
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'NLVR2', 'nlvr2'), prefix='test')

    def get_all_answers(self) -> list:
        return ['True', 'False']

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        assert len(row['images']) == 2
        assert len(row['options']) == 2
        gt = row['options']['AB'.index(row['answer'])][4:]
        assert gt in self.get_all_answers()
        prefix = '<image><image> Here is a statement about the images. Is it true or false?\n'
        assert row['question'].startswith(prefix)
        question = row['question'].removeprefix(prefix)
        return {
            'id': row['id'],
            'images': [img['bytes'] for img in row['images']],
            'question': question,
            'answer': gt,
            'template': "Please observe the two images carefully and determine whether the following statement is true or false base on them.\n{QUESTION}\nPlease answer True or False.\nYour answer is:"
        }

@register_benchmark('seed-bench')
class SEEDBench_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'SEED-Bench', 'data'))
        def has_image(image):
            return len(image) >= 0
        self.df = self.df.loc[self.df['image'].apply(has_image)]

        if data_args.max_num_images >= 0:
            filter = self.df['image'].apply(lambda imgs: len(imgs) <= data_args.max_num_images)
            if sum(filter) < len(self.df):
                logging.warning(f"[{data_args.bmk_name}] Dropped {len(self.df)-sum(filter)} samples with more than {data_args.max_num_images} images.")
                self.df = self.df[filter]

    def get_all_answers(self) -> list:
        return ['A', 'B', 'C', 'D']

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        return {
            'id': str(row['question_id']),
            'images': [row['image'][0]['bytes']],
            'question': row['question'],
            'options': [row['choice_a'], row['choice_b'], row['choice_c'], row['choice_d']],
            'answer': row['answer']
        }

@register_benchmark('scienceqa')
class ScienceQA_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer, subset):
        super().__init__(data_args, tokenizer)
        assert subset == 'train' or subset == 'test' or subset == 'validation', f"Unsupported subset: {subset}"
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'ScienceQA', 'ScienceQA-IMG'), prefix=subset)

    def get_all_answers(self) -> list:
        return ['A', 'B', 'C', 'D', 'E']

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        return {
            'id': str(i),
            'images': [row['image']['bytes']],
            'question': row['question'],
            'options': row['choices'],
            'answer': self.get_all_answers()[row['answer']]
        }

@register_benchmark('hallusion')
class Hallusion_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'HallusionBench'), prefix='image')

    def get_all_answers(self) -> list:
        options = list(set(self.df['gt_answer']))
        assert len(options) == 2
        return ['No', 'Yes']

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        return {
            'id': str(i),
            'images': [row['image']['bytes']],
            'question': row['question'],
            'answer': 'Yes' if int(row['gt_answer']) == 1 else 'No',
            'template': "Based on the content of the image, determine if the following statement is true.\n{QUESTION}\nPlease answer Yes or No.\nYour answer is:"
        }

@register_benchmark('mmstar')
class MMStar_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'MMStar'))
        self.spliters1 = ['\nOptions:', '\nChoices\n', '\nChoices:\n', '?\n(A) ']
        self.spliters2 = [
            re.compile(r'\s*([A-D]):\s*'),
            re.compile(r'\s*\(([A-D])\)\s*')
            ]

    def get_all_answers(self) -> list:
        return ['A', 'B', 'C', 'D']

    def __split_question_and_options(self, question):
        for i, spliter in enumerate(self.spliters1):
            if spliter in question:
                if i == 3:
                    cut_pt = question.index(spliter)
                    return question[:cut_pt + 1], question[cut_pt + 2:]
                return question.split(spliter)
        raise ValueError("Cannot split question and options")

    def __split_options(self, opt_str):
        for spliter in self.spliters2:
            cut_pts = []
            for i, m in enumerate(spliter.finditer(opt_str)):
                assert m.group(1) == self.get_all_answers()[i], opt_str
                cut_pts.append((m.start(), m.end()))
            if len(cut_pts) < 2 or cut_pts[0][0] != 0:
                continue
            options = []
            for i, cut in enumerate(cut_pts):
                next_cut = cut_pts[i+1][0] if i < len(cut_pts) - 1 else None
                opt = opt_str[cut[1]:next_cut]
                opt = opt.strip().strip(',')
                if not opt:
                    print(opt_str)
                options.append(opt)
            return options
        raise ValueError("Cannot split options")

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        question, opt_str = self.__split_question_and_options(row['question'])
        options = self.__split_options(opt_str)
        return {
            'id': str(row['index']),
            'images': [row['image']],
            'question': question,
            'options': options,
            'answer': row['answer']
        }

@register_benchmark('mmmu')
class MMMU_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer, subset):
        super().__init__(data_args, tokenizer)
        assert subset == 'dev' or subset == 'test' or subset == 'validation', f"Unsupported subset: {subset}"
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'MMMU', 'data'), prefix=subset)
        self.df['images'] = ''
        pattern_img = re.compile(r'<image [0-9]>')
        for i, row in self.df.iterrows():
            images = []
            for j in range(1, 8):
                if row[f'image_{j}']:
                    images.append(row[f'image_{j}']['bytes'])
            question = pattern_img.sub('<image>', row['question'])
            options = [pattern_img.sub('<image>', opt) for opt in eval(row['options'])]
            num_img_tags = question.count('<image>') + sum(opt.count('<image>') for opt in options)
            if len(images) > 1 and num_img_tags == 1 and '<image 1>' in row['question']:
                images = images[:1]
            if len(images) != num_img_tags:
                logging.warning(f"[{data_args.bmk_name}] Sample {row['id']} has mismatched number of images and tags, dropped")
                self.df.at[i, 'options'] = []
                continue
            self.df.at[i, 'question'] = question
            self.df.at[i, 'images'] = images
            self.df.at[i, 'options'] = options

        self.df.drop(columns=[f'image_{j}' for j in range(1, 8)], inplace=True)
        if subset == 'test':
            self.df.drop(columns=['answer'], inplace=True)
            self.default_answer = 'A'
        if data_args.max_num_images >= 0:
            filter = self.df['images'].apply(lambda imgs: len(imgs) <= data_args.max_num_images)
            if sum(filter) < len(self.df):
                logging.warning(f"[{data_args.bmk_name}] Dropped {len(self.df)-sum(filter)} samples with more than {data_args.max_num_images} images.")
                self.df = self.df[filter]
        filter = self.df['options'].apply(lambda opts: len(opts) >= 2 and len(opts) <= 26)
        if sum(filter) < len(self.df):
            logging.warning(f"[{data_args.bmk_name}] Dropped {len(self.df)-sum(filter)} samples with unsupported number of options.")
            self.df = self.df[filter]

    def get_all_answers(self) -> list:
        return list(string.ascii_uppercase[:max(self.df['options'].apply(len))])

    def _remove_image_tags(self, question: str, num_imgs) -> str:
        question = super()._remove_image_tags(question, num_imgs)
        if '<image>' in question:
            assert question.count('<image>') == num_imgs, question
            chunks = question.split('<image>')

            question = ""
            for i, c in enumerate(chunks[:-1]):
                question += c.removeprefix(')').removesuffix('(').strip()
                question += f" (Image {i+1}) "
            question += chunks[-1]
            question = question.replace('  (', ' (').replace(')  ', ') ')
            question = question.replace('  (', ' (').replace(')  ', ') ')
            question = question.lstrip('\n')
        return question

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            'id': row['id'],
            'images': row['images'],
            'question': row['question'],
            'options': row['options']
        }
        if 'answer' in row:
            sample['answer'] = row['answer']
        return sample

@register_benchmark('cmmmu')
class CMMMU_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer, subset):
        super().__init__(data_args, tokenizer)
        assert subset == 'dev' or subset == 'test' or subset == 'val', f"Unsupported subset: {subset}"
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'CMMMU', 'data'), prefix=subset)
        self.df['images'] = ''
        self.df['options'] = ''
        pattern_img = re.compile(r'<img.*?>')
        for i, row in self.df.iterrows():
            img_tags = set()
            question = row['question']
            while True:
                m = pattern_img.search(question)
                if not m:
                    break
                if m.group(0) in img_tags:
                    question = question[:m.start()] + '图' + question[m.end():]
                else:
                    img_tags.add(m.group(0))
                    question = question[:m.start()] + '<image>' + question[m.end():]
            self.df.at[i, 'question'] = question

            options = []
            if row['type'] == '选择':
                if subset != 'test' and row['answer'] not in list('ABCD'):
                    self.df.at[i, 'type'] = '多选'
                    logging.warning(f"[{data_args.bmk_name}] Sample {row['id']} has multi-choice: {row['answer']}, dropped")
                    continue
                for j in range(1, 5):
                    if row[f'option{j}']:
                        options.append(pattern_img.sub('<image>', row[f'option{j}']))
                    else:
                        break
                assert len(options) == 4, row['options']
            elif row['type'] == '判断':
                assert len(options) == 0
                assert subset != 'test' and row['answer'] in list('对错')
                options.append('对')
                options.append('错')
                self.df.at[i, 'answer'] = 'AB'['对错'.index(row['answer'])]
            else:
                continue
            self.df.at[i, 'options'] = options

            images = []
            for j in range(1, 6):
                if row[f'image_{j}']:
                    images.append(row[f'image_{j}']['bytes'])
                else:
                    break
            self.df.at[i, 'images'] = images

            num_tags = question.count('<image>') + sum(opt.count('<image>') for opt in options)
            if num_tags != len(images):
                self.df.at[i, 'type'] = 'bad'
                logging.warning(f"[{data_args.bmk_name}] Sample {row['id']} has mismatched number of images and tags, dropped")

        self.df.drop(columns=[f'image_{j}' for j in range(1, 6)], inplace=True)
        if subset == 'test':
            self.df.drop(columns=['answer'], inplace=True)
            self.default_answer = 'A'

        if data_args.max_num_images >= 0:
            filter = self.df['images'].apply(lambda imgs: len(imgs) <= data_args.max_num_images)
            if sum(filter) < len(self.df):
                logging.warning(f"[{data_args.bmk_name}] Dropped {len(self.df)-sum(filter)} samples with more than {data_args.max_num_images} images.")
                self.df = self.df[filter]

        filter = self.df['type'].apply(lambda ty: ty == '选择' or ty == '判断')
        if sum(filter) < len(self.df):
            logging.warning(f"[{data_args.bmk_name}] Dropped {len(self.df)-sum(filter)} samples with unsupported question type.")
            self.df = self.df[filter]

    def get_all_answers(self) -> list:
        return list(string.ascii_uppercase[:max(self.df['options'].apply(len))])

    def _format_opt_caps(self, sample) -> str:
        return super()._format_opt_caps(sample, 'zh')

    def _remove_image_tags(self, question: str, num_imgs) -> str:
        assert question.count('<image>') == num_imgs, f"{num_imgs} {question}"
        question = super()._remove_image_tags(question, num_imgs)
        if '<image>' in question:
            chunks = question.split('<image>')
            question = ""
            for i, c in enumerate(chunks[:-1]):
                question += c.removeprefix(')').removesuffix('(').strip()
                question += f" (Image {i+1}) "
            question += chunks[-1]
            question = question.replace('  (', ' (').replace(')  ', ') ')
            question = question.replace('  (', ' (').replace(')  ', ') ')
            question = question.lstrip('\n')
        return question

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            'id': row['id'],
            'images': row['images'],
            'question': row['question'],
            'options': row['options'],
            'template': "{QUESTION}\n{OPTIONS}\n\n" \
                "请选择正确答案并写出答案前的标号 ({OPT_CAPS}）。\n你的答案是:"
            }

        if 'answer' in row:
            sample['answer'] = row['answer']
        return sample

@register_benchmark('mmlu')
class MMLU_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer, subset):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'mmlu', 'data', subset),
                                format='csv', header=None, names=['Question', 'A', 'B', 'C', 'D', 'Answer'])

    def get_all_answers(self) -> list:
        return list('ABCD')

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        assert row['Answer'] in self.get_all_answers(), row['Answer']
        sample = {
            'id': str(i),
            'question': row['Question'],
            'options': [row['A'], row['B'], row['C'], row['D']],
            'answer': row['Answer']
        }
        return sample

@register_benchmark('cmmlu')
class CMMLU_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer, subset):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'cmmlu', 'data', subset),
                                format='csv', header=0, names=['id', 'Question', 'A', 'B', 'C', 'D', 'Answer'])

    def get_all_answers(self) -> list:
        return list('ABCD')

    def _format_opt_caps(self, sample) -> str:
        return super()._format_opt_caps(sample, 'zh')

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            'id': str(i),
            'question': row['Question'],
            'options': [row['A'], row['B'], row['C'], row['D']],
            'answer': row['Answer'],
            'template': "{QUESTION}\n{OPTIONS}\n\n" \
                "请选择正确答案并写出答案前的标号 ({OPT_CAPS}）。\n你的答案是:"
        }
        return sample

@register_benchmark('mantis')
class Mantis_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'Mantis-Eval', 'mantis_eval'))
        opts_yes_no = ['Yes', 'No']

        pattern_opt_caps = [
            re.compile(r'^([A-E]):\s*'),
            re.compile(r'^\(([A-E])\)\s*')
            ]
        for i, row in self.df.iterrows():
            if row['question_type'] == 'short-answer':
                if row['answer'] == 'Yes' or row['answer'] == 'No':
                    self.df.at[i, 'options'] = opts_yes_no
                    yesno2optid = int(row['answer'] == 'No') # Yes:0, No:1
                    self.df.at[i, 'answer'] = 'AB'[yesno2optid]
            else:
                options = {}
                for opt in row['options']:
                    for pat in pattern_opt_caps:
                        m = pat.search(opt)
                        if m:
                            break
                    if not m:
                        break
                    options[m.group(1)] = opt[m.end():]
                opt_caps = sorted(list(options.keys()))
                if len(options) != len(row['options']) or opt_caps != self.get_all_answers()[:len(opt_caps)]:
                    self.df.at[i, 'options'] = []
                self.df.at[i, 'options'] = [options[cap] for cap in opt_caps]

            num_tags = row['question'].count('<image>')
            num_tags += sum([opt.count('<image>') for opt in row['options']])
            if num_tags != len(row['images']):
                if num_tags == 0:
                    img_tag = ''
                    for i_img in range(len(row['images'])):
                        img_tag += f"Image {i_img}: <image>\n"
                    self.df.at[i, 'question'] = img_tag + row['question']
                else:
                    self.df.at[i, 'options'] = []

        filter = self.df['options'].apply(lambda opts: len(opts) >= 2 and len(opts) <= 5)
        if sum(filter) < len(self.df):
            print(f"[{data_args.bmk_name}] Dropped {len(self.df)-sum(filter)} samples with unsupported number of options.")
            self.df = self.df[filter]

        filter = self.df['answer'].apply(lambda answer: answer in self.get_all_answers())
        if sum(filter) < len(self.df):
            print.warning(f"[{data_args.bmk_name}] Dropped {len(self.df)-sum(filter)} samples with unsupported answer.")
            self.df = self.df[filter]

    def get_all_answers(self) -> list:
        return list('ABCDE')

    def _remove_image_tags(self, question: str, num_imgs) -> str:
        assert question.count('<image>') == num_imgs, question
        chunks = question.split('<image>')
        question = ""
        for i, c in enumerate(chunks[:-1]):
            question += c.removeprefix(')').removesuffix('(').strip()
            question += f" (image {i+1}) " if num_imgs > 1 else " (the image) "
        question += chunks[-1]
        question = question.replace('  (', ' (').replace(')  ', ') ')
        question = question.replace('  (', ' (').replace(')  ', ') ')
        question = question.lstrip('\n')
        return question

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            'id': str(i),
            'question': row['question'],
            'images': [image['bytes'] for image in row['images']],
            'options': row['options'],
            'answer': row['answer']
        }
        return sample

@register_benchmark('q-bench2')
class QBench2_Dataset(BenchmarkDataset):
    def __init__(self, data_args: BenchmarkDataArguments, tokenizer: PTTokenizer, subset):
        super().__init__(data_args, tokenizer)
        self.df = load_bmk_data(os.path.join(data_args.bmk_root, 'Q-Bench2-HF', 'data'), prefix=subset)
        opt_cols = [f'option{i}' for i in range(4)]
        self.df['options'] = ''
        for i, row in self.df.iterrows():
            options = []
            for opt in opt_cols:
                opt = row[opt]
                if opt == 'N/A':
                    break
                options.append(opt)
            self.df.at[i, 'options'] = options
        if subset == 'test':
            opt_cols.append('correct_choice')
            self.default_answer = 'A'
        self.df.drop(columns=opt_cols, inplace=True)

    def get_all_answers(self) -> list:
        return list('ABCD')

    def get_sample(self, i: int) -> Dict:
        row = self.df.iloc[i]
        sample = {
            'id': str(row['id']),
            'question': row['question'],
            'images': [row['image1']['bytes'], row['image2']['bytes']],
            'options': row['options']
        }
        if 'correct_choice' in row:
            sample['answer'] = row['correct_choice']
        return sample
