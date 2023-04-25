# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from enum import Enum
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.datasets as datasets
from torch import distributed, nn
from torch.nn.functional import normalize, one_hot, softmax
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.utilities.data import dim_zero_cat, select_topk

import unicom

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
distributed.init_process_group("nccl")

parser = argparse.ArgumentParser()
parser.add_argument("--train-dataset", type=str, help="Training dataset",)
parser.add_argument("--val-dataset", type=str, help="Validation dataset",)
parser.add_argument("--nb_knn", nargs="+", type=int, help="Number of NN to use. 20 is usually working the best.",)
parser.add_argument("--temperature", type=float, help="Temperature used in the voting coefficient",)
parser.add_argument("--gather-on-cpu", action="store_true",
    help="Whether to gather the train features on cpu, slower"
    "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
)
parser.add_argument("--batch-size", type=int, help="Batch size.",)
parser.add_argument("--model-name", default="ViT-B/32",)
parser.add_argument("--n-per-class-list", nargs="+", type=int, help="Number to take per class",)
parser.add_argument("--n-tries", type=int, help="Number of tries",)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--output-dir", default="output",)
parser.set_defaults(
    train_dataset="ImageNet:split=TRAIN",
    val_dataset="ImageNet:split=VAL",
    nb_knn=[10, 20, 100, 200],
    temperature=0.07,
    batch_size=256,
    model_name="ViT-B/32",
    n_per_class_list=[-1],
    n_tries=1,
)
args = parser.parse_args()

@torch.no_grad()
def main():
    
    model, transform = unicom.load(args.model_name)
    model = ModelWithNormalize(model)
    model = model.cuda()

    output_dir = args.output_dir
    nb_knn = args.nb_knn
    temperature = args.temperature

    accuracy_averaging = AccuracyAveraging.MEAN_ACCURACY
    gather_on_cpu = args.gather_on_cpu
    batch_size = args.batch_size
    num_workers = args.num_workers
    n_per_class_list = args.n_per_class_list
    n_tries = args.n_tries

    train_dataset = datasets.ImageFolder(args.train_dataset, transform)
    val_dataset = datasets.ImageFolder(args.val_dataset, transform)


    results_dict_knn = eval_knn(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        accuracy_averaging=accuracy_averaging,
        nb_knn=nb_knn,
        temperature=temperature,
        batch_size=batch_size,
        num_workers=num_workers,
        gather_on_cpu=gather_on_cpu,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
    )

    results_dict = {}
    if rank == 0:
        for knn_ in results_dict_knn.keys():
            top1 = results_dict_knn[knn_]["top-1"].item() * 100.0
            top5 = results_dict_knn[knn_]["top-5"].item() * 100.0
            results_dict[f"{knn_} Top 1"] = top1
            results_dict[f"{knn_} Top 5"] = top5
            print(f"{knn_} classifier result: Top1: {top1:.2f} Top5: {top5:.2f}")

    metrics_file_path = os.path.join(output_dir, "results_eval_knn.json")
    with open(metrics_file_path, "a") as f:
        for k, v in results_dict.items():
            f.write(json.dumps({k: v}) + "\n")

    if distributed.is_initialized():
        torch.distributed.barrier()
    # return results_dict


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return normalize(self.model(x))


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value


class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features
    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, train_features, train_labels, 
                 nb_knn, T, local_rank=0, num_classes=1000, rank=0, world_size=1):
        super().__init__()

        self.rank = rank
        self.world_size = world_size

        self.local_rank = local_rank
        self.train_features_rank_T = train_features.chunk(world_size)[rank].T.cuda()
        self.candidates = train_labels.chunk(world_size)[rank].view(1, -1).cuda()

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).cuda()
        torch.distributed.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        if self.rank != source_rank:
            broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype).cuda()
        torch.distributed.broadcast(broadcasted, source_rank)

        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.world_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.world_size)]

        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)

        if self.rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.world_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


def create_module_dict(*, module, n_per_class_list, n_tries, nb_knn, train_features, train_labels):
    modules = {}
    mapping = create_class_indices_mapping(train_labels)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            full_module = module(
                train_features=train_features,
                train_labels=train_labels,
                nb_knn=nb_knn,
            )
            modules["full"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):
            final_indices = filter_train(mapping, npc, seed=t)
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc])
            all_tries[str(t)] = module(
                train_features=train_features[final_indices],
                train_labels=train_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


def filter_train(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))}
    return mapping


class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}


def eval_knn(
    model,
    train_dataset,
    val_dataset,
    accuracy_averaging,
    nb_knn,
    temperature,
    batch_size,
    num_workers,
    gather_on_cpu,
    n_per_class_list=[-1],
    n_tries=1,
):

    print("Extracting features for train set...")
    train_features, train_labels = extract_feat(model, train_dataset, batch_size, num_workers)
    print(f"Train features created, shape {train_features.shape}.")

    n_data = len(val_dataset)
    idx_all_rank = list(range(n_data))
    num_local = n_data // world_size + int(rank < n_data % world_size)
    start = n_data // world_size * rank + min(rank, n_data % world_size)
    idx_this_rank = idx_all_rank[start:start + num_local]
    dataset_this_rank = Subset(val_dataset, idx_this_rank)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": False
    }
    val_dataloader = DataLoader(dataset_this_rank, **kwargs)

    num_classes = train_labels.max() + 1
    metric_collection = build_topk_accuracy_metric(accuracy_averaging, num_classes=num_classes)

    # device = torch.cuda.current_device()
    # partial_module = partial(KnnModule, T=temperature, device=device, num_classes=num_classes)
    partial_module = partial(KnnModule, T=temperature, num_classes=num_classes, rank=rank, world_size=world_size)
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
        nb_knn=nb_knn,
        train_features=train_features,
        train_labels=train_labels,
    )
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items():
        for t, knn_try in knn_module.items():
            postprocessors = {
                **postprocessors,
                **{(n_per_class, t, k): DictKeysModule([n_per_class, t, k]) for k in knn_try.nb_knn},
            }
            metrics = {**metrics, **{(n_per_class, t, k): metric_collection.clone() for k in knn_try.nb_knn}}
    model_with_knn = torch.nn.Sequential(model, knn_module_dict)

    # ============ evaluation ... ============
    print("Start the k-NN classification.")
    results_dict = evaluate(model_with_knn, val_dataloader, postprocessors, metrics)

    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[(n_per_class, first_try, k)].keys()  # keys are e.g. `top-1` and `top-5`
            results_dict[(n_per_class, k)] = {
                key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
                for key in keys
            }
            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]

    return results_dict



class MetricType(Enum):
    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    PER_CLASS_ACCURACY = "per_class_accuracy"
    IMAGENET_REAL_ACCURACY = "imagenet_real_accuracy"

    @property
    def accuracy_averaging(self):
        return getattr(AccuracyAveraging, self.name, None)

    def __str__(self):
        return self.value

def build_metric(metric_type: MetricType, *, num_classes: int, ks: Optional[tuple] = None):
    if metric_type.accuracy_averaging is not None:
        return build_topk_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )
    elif metric_type == MetricType.IMAGENET_REAL_ACCURACY:
        return build_topk_imagenet_real_accuracy_metric(
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )

    raise ValueError(f"Unknown metric type {metric_type}")


def build_topk_imagenet_real_accuracy_metric(num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {f"top-{k}": ImageNetReaLAccuracy(top_k=k, num_classes=int(num_classes)) for k in ks}
    return MetricCollection(metrics)


def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks
    }
    return MetricCollection(metrics)

class ImageNetReaLAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.add_state("tp", [], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        # preds [B, D]
        # target [B, A]
        # preds_oh [B, D] with 0 and 1
        # select top K highest probabilities, use one hot representation
        preds_oh = select_topk(preds, self.top_k)
        # target_oh [B, D + 1] with 0 and 1
        target_oh = torch.zeros((preds_oh.shape[0], preds_oh.shape[1] + 1), device=target.device, dtype=torch.int32)
        target = target.long()
        # for undefined targets (-1) use a fake value `num_classes`
        target[target == -1] = self.num_classes
        # fill targets, use one hot representation
        target_oh.scatter_(1, target, 1)
        # target_oh [B, D] (remove the fake target at index `num_classes`)
        target_oh = target_oh[:, :-1]
        # tp [B] with 0 and 1
        tp = (preds_oh * target_oh == 1).sum(dim=1)
        # at least one match between prediction and target
        tp.clip_(max=1)
        # ignore instances where no targets are defined
        mask = target_oh.sum(dim=1) > 0
        tp = tp[mask]
        self.tp.append(tp)  # type: ignore

    def compute(self) -> torch.Tensor:
        tp = dim_zero_cat(self.tp)  # type: ignore
        return tp.float().mean()


def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.cuda()

    for samples, targets in data_loader:
        outputs = model(samples.cuda())
        targets = targets.cuda()

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)
    stats = {k: metric.compute() for k, metric in metrics.items()}

    return stats


def extract_feat(
        model: torch.nn.Module,
        dataset: Dataset,
        batch_size: int,
        num_workers: int) -> Tuple[torch.Tensor, torch.Tensor]:
    model.cuda()
    model.eval()
    n_data = len(dataset)
    idx_all_rank = list(range(n_data))
    num_local = n_data // world_size + int(rank < n_data % world_size)
    start = n_data // world_size * rank + min(rank, n_data % world_size)
    idx_this_rank = idx_all_rank[start:start + num_local]
    dataset_this_rank = Subset(dataset, idx_this_rank)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": False
    }
    dataloader = DataLoader(dataset_this_rank, **kwargs)
    x = None
    y_np = []
    idx = 0
    for image, label in dataloader:
        image = image.cuda()

        embedding = model(image)
        # print(embedding.size())
        embedding_size: int = embedding.size(1)
        if x is None:
            size = [len(dataset_this_rank), embedding_size]
            x = torch.zeros(*size, device=image.device)
        x[idx:idx + embedding.size(0)] = embedding
        y_np.append(np.array(label))
        idx += embedding.size(0)
    x = x.cpu()
    y_np = np.concatenate(y_np, axis=0)

    if distributed.is_initialized():
        gather_list_x = [None for i in range(world_size)]
        gather_list_y = [None for i in range(world_size)]
        distributed.all_gather_object(gather_list_x, x)
        distributed.all_gather_object(gather_list_y, y_np)
        x = torch.cat(gather_list_x, dim=0)
        y_np = np.concatenate(gather_list_y, axis=0)

    return x, torch.from_numpy(y_np)

if __name__ == "__main__":
    main()
