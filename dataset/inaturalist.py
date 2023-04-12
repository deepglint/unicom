# https://github.com/elias-ramzi/ROADMAP/blob/main/roadmap/datasets/base_dataset.py
# https://github.com/elias-ramzi/ROADMAP/blob/main/roadmap/datasets/inaturalist.py

import json
from collections import Counter
from os.path import join

import torch
from PIL import Image
import os
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(
        self,
        multi_crop=False,
        size_crops=[224, 96],
        nmb_crops=[2, 6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1., 0.14],
        size_dataset=-1,
        return_label='none',
    ):
        super().__init__()

        if not multi_crop:
            self.get_fn = self.simple_get
        else:
            raise

    def __len__(self,):
        return len(self.paths)

    @property
    def my_at_R(self,):
        if not hasattr(self, '_at_R'):
            self._at_R = max(Counter(self.labels).values())
        return self._at_R

    def get_instance_dict(self,):
        self.instance_dict = {cl: [] for cl in set(self.labels)}
        for idx, cl in enumerate(self.labels):
            self.instance_dict[cl].append(idx)

    def get_super_dict(self,):
        if hasattr(self, 'super_labels') and self.super_labels is not None:
            self.super_dict = {ct: {} for ct in set(self.super_labels)}
            for idx, cl, ct in zip(range(len(self.labels)), self.labels, self.super_labels):
                try:
                    self.super_dict[ct][cl].append(idx)
                except KeyError:
                    self.super_dict[ct][cl] = [idx]

    def simple_get(self, idx):
        pth = self.paths[idx]
        img = Image.open(pth).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        label = torch.tensor([label])
        out = {"image": img, "label": label, "path": pth}

        if hasattr(self, 'super_labels') and self.super_labels is not None:
            super_label = self.super_labels[idx]
            super_label = torch.tensor([super_label])
            out['super_label'] = super_label

        return out

    def multiple_crop_get(self, idx):
        pth = self.paths[idx]
        image = Image.open(pth).convert('RGB')
        multi_crops = list(map(lambda trans: trans(image), self.trans))

        if self.return_label == 'real':
            label = self.labels[idx]
            labels = [label] * len(multi_crops)
            return {"image": multi_crops, "label": labels, "path": pth}

        if self.return_label == 'hash':
            label = abs(hash(pth))
            labels = [label] * len(multi_crops)
            return {"image": multi_crops, "label": labels, "path": pth}

        return {"image": multi_crops, "path": pth}

    def __getitem__(self, idx):
        return self.get_fn(idx)

    def __repr__(self,):
        return f"{self.__class__.__name__}(mode={self.mode}, len={len(self)})"


class INaturalistDataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            mode = ['train']
        elif mode == 'test':
            mode = ['test']
        elif mode == 'all':
            mode = ['train', 'test']
        else:
            raise ValueError(f"Mode unrecognized {mode}")

        self.paths = []
        for splt in mode:
            with open(join(self.data_dir, f'Inat_dataset_splits/Inaturalist_{splt}_set1.txt')) as f:
                paths = f.read().split("\n")
                paths.remove("")
            self.paths.extend([join(self.data_dir, pth) for pth in paths])

        with open(join(self.data_dir, 'train2018.json')) as f:
            db = json.load(f)['categories']
            self.db = {}
            for x in db:
                _ = x.pop("name")
                id_ = x.pop("id")
                x["species"] = id_
                self.db[id_] = x

        self.labels_name = [int(x.split("/")[-2]) for x in self.paths]
        self.labels_to_id = {cl: i for i, cl in enumerate(sorted(set(self.labels_name)))}
        self.labels = [self.labels_to_id[x] for x in self.labels_name]

        self.hierarchy_name = {}
        for x in self.labels_name:
            for key, val in self.db[x].items():
                try:
                    self.hierarchy_name[key].append(val)
                except KeyError:
                    self.hierarchy_name[key] = [val]

        self.hierarchy_name_to_id = {}
        self.hierarchy_labels = {}
        for key, lst in self.hierarchy_name.items():
            self.hierarchy_name_to_id[key] = {cl: i for i, cl in enumerate(sorted(set(lst)))}
            self.hierarchy_labels[key] = [self.hierarchy_name_to_id[key][x] for x in lst]

        self.super_labels_name = [x.split("/")[-3] for x in self.paths]
        self.super_labels_to_id = {scl: i for i, scl in enumerate(sorted(set(self.super_labels_name)))}
        self.super_labels = [self.super_labels_to_id[x] for x in self.super_labels_name]

        self.get_instance_dict()
        self.get_super_dict()

    def get_super_dict(self,):
        if hasattr(self, 'super_labels') and self.super_labels is not None:
            self.super_dict = {ct: {} for ct in set(self.super_labels)}
            for idx, cl, ct in zip(range(len(self.labels)), self.labels, self.super_labels):
                try:
                    self.super_dict[ct][cl].append(idx)
                except KeyError:
                    self.super_dict[ct][cl] = [idx]

    def get_instance_dict(self,):
        self.instance_dict = {cl: [] for cl in set(self.labels)}
        for idx, cl in enumerate(self.labels):
            self.instance_dict[cl].append(idx)

    def __len__(self,):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.simple_get(idx)

    def simple_get(self, idx):
        pth = self.paths[idx]
        img = Image.open(pth).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label


def get_testset(root, transform):
    return INaturalistDataset(os.path.join(root, "iNaturalist"), "test", transform)


def get_trainset(root, transform):
    return INaturalistDataset(os.path.join(root, "iNaturalist"), "train", transform)
