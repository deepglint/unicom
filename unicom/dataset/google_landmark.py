import os

import PIL.Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

num_classes = 81314


class GoogleLandmarkDataset(Dataset):
    def __init__(self, split="train", transform=None, usage="Public", root="data/GLDv2/"):
        super().__init__()
        assert split in ["train", "test", "index"]
        assert usage in ["Public", "Private"]
        self.split = split
        self.transform = transform
        self.dataset_root = os.path.join(root, self.split)

        if self.split == "train":
            raise
        elif self.split == "test":
            df = pd.read_csv(
                os.path.join(self.dataset_root, "retrieval_solution_v2.1.csv"))
            self.numpy_id_images_Usage: np.ndarray = df.loc[df["Usage"] == usage].values

        elif self.split == "index":
            df = pd.read_csv(
                os.path.join(self.dataset_root, "index.csv")
            )
            self.numpy_id: np.ndarray = df.values


    def __len__(self,):
        if self.split == "test":
            return len(self.numpy_id_images_Usage)
        elif self.split == "index":
            return len(self.numpy_id)
        else:
            return 4000000

    def __getitem__(self, idx):
        if self.split == "test":
            line = self.numpy_id_images_Usage[idx]
            id_, images, usage = line[0], line[1], line[2]
            # images = images.split(" ")
            image_file = os.path.join(self.dataset_root, id_[0], id_[1], id_[2], f"{id_}.jpg")
            image = PIL.Image.open(image_file).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return (image, images)
        elif self.split == "index":
            line = self.numpy_id[idx]
            id_ = line[0]
            image_file = os.path.join(self.dataset_root, id_[0], id_[1], id_[2], f"{id_}.jpg")
            image = PIL.Image.open(image_file).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return (image, id_)
        else:
            return (None, None)


def get_testset_query_val(transform):
    return GoogleLandmarkDataset("test", transform, "Public")

def get_testset_query_test(transform):
    return GoogleLandmarkDataset("test", transform, "Private")

def get_testset_gallery(transform):
    return GoogleLandmarkDataset("index", transform, "Public")
