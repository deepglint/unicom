import argparse
import math
import os
import random
import time
from functools import partial
from typing import Callable, Dict, Tuple

import numpy as np
import PIL
import torch
from torch import distributed, optim
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Subset
from torchvision import transforms

import unicom
from partial_fc import CombinedMarginLoss, PartialFC_V2

parser = argparse.ArgumentParser(
    description="retrieval is a command-line tool that provides functionality for fine-tuning the Unicom model on retrieval tasks. With this tool, you can easily adjust the unicom model to achieve optimal performance on a variety of image retrieval tasks. Simply specify the task-specific parameters and let the tool handle the rest.")
parser.add_argument("--batch_size", default=128, type=int, help="The batch size to use for training and inference.")
parser.add_argument("--dataset", default="cub", help="The dataset to load for training and evaluation.")
parser.add_argument("--debug", default=0, type=int, help="A flag indicating whether to run the code in debug mode (with additional logging or other debugging aids).")
parser.add_argument("--epochs", type=int, default=32, help="The number of epochs to train the model for.")
parser.add_argument("--eval", action="store_true", help="A flag indicating whether to run model evaluation after training.")
parser.add_argument("--lr", type=float, default=0.0001, help="The learning rate to use for training the model.")
parser.add_argument("--lr_pfc_weight", type=float, default=5.0, help="The weight to apply to the learning rate for the Partial FC layer during training. Sure, when fine-tuning a pre-trained neural network, it is usually recommended to adjust the learning rates of different layers in order to achieve better performance. For example, the learning rate of the backbone layers (i.e., the pre-trained layers) should be set lower because they already have learned features, while the learning rate of the Partial FC layer should be set higher, as it needs to adapt to the new task.")
parser.add_argument("--input_size", default=224, type=int, help="The size of the input images for the model.")
parser.add_argument("--gradient_acc", default=1, type=int, help="The number of times gradients are accumulated before updating the model's parameters.")
parser.add_argument("--model_name", default="ViT-B/16", help="The name of the pre-trained model to use for feature extraction.")
parser.add_argument("--margin_loss_m1", type=float, default=1.0, help="The margin parameter (m1) for the margin loss function.")
parser.add_argument("--margin_loss_m2", type=float, default=0.3, help="The margin parameter (m1) for the margin loss function.")
parser.add_argument("--margin_loss_m3", type=float, default=0.0, help="The margin parameter (m3) for the margin loss function.")
parser.add_argument("--margin_loss_s", type=float, default=32, help="The scale parameter (s) for the margin loss function.")
parser.add_argument("--margin_loss_filter", type=float, default=0.0, help="The filter parameter for the margin loss function.")
parser.add_argument("--num_workers", default=8, type=int, help="The number of workers to use for data loading.")
parser.add_argument("--num_feat", default=None, type=int, help="This parameter is used to set the dimensionality of the features sampled for use in model training and evaluation. ")
parser.add_argument("--optimizer", default="adamw", help="The optimizer to use for the training process, default is AdamW.")
parser.add_argument("--output_dim", type=int, default=768, help="The desired dimensionality of the output embeddings in ViT.")
parser.add_argument("--output", default="/tmp/tmp_for_training", help="")
parser.add_argument("--resume", default="NULL", help="The path to a saved checkpoint to resume training from.")
parser.add_argument("--sample_rate", default=1.0, type=float, help="The negative sample rate to be used for partial FC. It helps to reduce memory usage, increase training speed And can significantly improve performance on datasets with high levels of noise")
parser.add_argument("--seed", type=int, default=1024, help="The random seed to use for reproducibility.")
parser.add_argument("--transform", default=None, type=str, help="Transofrm in pytorch dataloader.")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay.")
parser.add_argument("--color_jitter", type=float, default=0.4, help="The amount of color jittering to apply during data augmentation.")
parser.add_argument("--aa", type=str, default='rand-m9-mstd0.5-inc1', help="The amount of color jittering to apply during data augmentation. The default value is 'rand-m9-mstd0.5-inc1'. ")
parser.add_argument("--reprob", type=float, default=0.25, help="The probability of replacing pixels during training using CutOut.")
parser.add_argument("--remode", type=str, default="pixel", help="The mode of replacement to use during training when using CutOut.")
parser.add_argument("--recount", type=int, default=1, help="")


args = parser.parse_args()

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)


def get_dataset(dataset_name: str, transform: Callable, transform_train=None) -> Dict:
    if transform_train is None:
        transform_train = transform
    root = "data"

    if dataset_name == "sop":
        from dataset import SOP
        trainset = SOP(root, "train", transform_train)
        testset = SOP(root, "eval", transform)
        trainset.num_classes = trainset.nb_classes()
        return {"train": trainset, "test": testset, "metric": "rank1"}

    elif dataset_name == "cub":
        from dataset import CUBirds
        trainset = CUBirds(root, "train", transform_train)
        testset = CUBirds(root, "eval", transform)
        trainset.num_classes = trainset.nb_classes()
        return {"train": trainset, "test": testset, "metric": "rank1"}

    elif dataset_name == "car":
        from dataset import Cars
        trainset = Cars(root, "train", transform_train)
        testset = Cars(root, "eval", transform)
        trainset.num_classes = trainset.nb_classes()
        return {"train": trainset, "test": testset, "metric": "rank1"}

    elif dataset_name == "inshop":
        from dataset import Inshop_Dataset
        trainset = Inshop_Dataset(root, "train", transform_train)
        query = Inshop_Dataset(root, "query", transform)
        gallery = Inshop_Dataset(root, "gallery", transform)
        trainset.num_classes = trainset.nb_classes()
        return {"train": trainset, "query": query, "gallery": gallery, "metric": "rank1"}
    elif dataset_name == "inat":
        from dataset import inaturalist
        trainset = inaturalist.get_trainset(root, transform_train)
        testset = inaturalist.get_testset(root, transform)
        trainset.num_classes = 5690
        return {"train": trainset, "test": testset, "metric": "rank1"}
    else:
        raise


class WarpModule(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self,  x):
        return self.model(x)


def main():
    if world_size >= 1:
        distributed.init_process_group(backend="nccl")

    if args.eval:
        model, transform_clip = unicom.load(args.model_name)
        model = model.cuda()
        model = WarpModule(model)
        dataset_dict: Dict = get_dataset(args.dataset, transform_clip)
        score = evaluation(model, dataset_dict,
                           args.batch_size, args.num_workers)
        if rank == 0:
            if isinstance(score, Tuple):
                for i in score:
                    print(i, end=",")
            else:
                print(score, end=",")
    else:
        if rank == 0:
            for arg in vars(args):
                print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
        if args.model_name == "ViT-L/14@336px":
            transform_train = get_transform(336)
            transform_test = get_transform(336, is_train=False)
        else:
            transform_train = get_transform(224)
            transform_test = get_transform(224, is_train=False)

        model, transform_clip = unicom.load(args.model_name)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = WarpModule(model)
        if args.transform == "origin_clip":
            transform_train = transform_test = transform_clip
        dataset_dict: Dict = get_dataset(
            args.dataset, transform_test, transform_train)
        dataset_train = dataset_dict['train']
        model.train()
        model.cuda()
        backbone = torch.nn.parallel.DistributedDataParallel(
            module=model,
            bucket_cap_mb=32,
            find_unused_parameters=True,
            static_graph=True)
        margin_loss = CombinedMarginLoss(
            args.margin_loss_s,
            args.margin_loss_m1,
            args.margin_loss_m2,
            args.margin_loss_m3,
            args.margin_loss_filter
        )
        if args.optimizer == "adamw":
            module_partial_fc = PartialFC_V2(
                margin_loss, args.output_dim, dataset_train.num_classes,
                args.sample_rate, False, sample_num_feat=args.num_feat)
            module_partial_fc.train().cuda()
            opt = torch.optim.AdamW(
                params=[
                    {"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters(), "lr": args.lr * args.lr_pfc_weight}],
                lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            module_partial_fc = PartialFC_V2(
                margin_loss, args.output_dim, dataset_train.num_classes,
                args.sample_rate, False, sample_num_feat=args.num_feat)
            module_partial_fc.train().cuda()
            opt = torch.optim.SGD(
                params=[
                    {"params": backbone.parameters()},
                    {"params": module_partial_fc.parameters(), "lr": args.lr * args.lr_pfc_weight}],
                lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"{args.optimizer} is wrong")

        num_train_set = len(dataset_train)
        train_sampler = DistributedSampler(
            dataset_train, num_replicas=world_size,
            rank=rank, shuffle=True)
        init_fn = partial(
            worker_init_fn, num_workers=args.num_workers,
            rank=rank, seed=args.seed)
        loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            worker_init_fn=init_fn,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,)

        steps_per_epoch = num_train_set // world_size // args.batch_size + 1
        steps_total = args.epochs * steps_per_epoch

        args.lr_scheduler = "cosine"
        if args.lr_scheduler == "cosine":
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                max_lr=[args.lr, args.lr * args.lr_pfc_weight],
                steps_per_epoch=steps_per_epoch,
                epochs=args.epochs,
                pct_start=0.1,
            )
        elif args.lr_scheduler == "linear":
            lr_scheduler = optim.lr_scheduler.LinearLR(
                optimizer=opt, start_factor=1.0, end_factor=0.0,
                total_iters=args.epochs * steps_per_epoch)
        else:
            raise

        callback_func = SpeedCallBack(10, steps_total, args.batch_size)
        auto_scaler = torch.cuda.amp.grad_scaler.GradScaler(
            growth_interval=200)
        global_step = 0
        max_score = 0
        for epoch in range(0, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for _, (img, local_labels) in enumerate(loader_train):
                img = img.cuda()
                local_labels = local_labels.long().cuda()
                with torch.cuda.amp.autocast(False):
                    local_embeddings = backbone(img)
                local_embeddings.float()

                local_labels = local_labels.cuda()
                loss = module_partial_fc(local_embeddings, local_labels)
                auto_scaler.scale(loss).backward()

                if global_step % args.gradient_acc == 0:
                    auto_scaler.step(opt)
                    auto_scaler.update()
                    opt.zero_grad()

                lr_scheduler.step()
                global_step += 1

                with torch.no_grad():
                    callback_func(
                        lr_scheduler,
                        float(loss),
                        global_step,
                        auto_scaler.get_scale())
            score = evaluation(model, dataset_dict,
                               args.batch_size, num_workers=args.num_workers)
            if isinstance(score, float):
                if score > max_score:
                    max_score = score
            if rank == 0:
                print(f"eval result is {max_score}, epoch is {epoch}")
            model.train()


class SpeedCallBack(object):
    def __init__(self, frequent, steps_total, batch_size):
        self.batch_size = batch_size
        self.frequent = frequent
        self.steps_total = steps_total
        self.loss_metric = AverageMeter()
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.time_start = time.time()
        self.init = False
        self.tic = 0

    def __call__(
            self,
            lr_scheduler: optim.lr_scheduler._LRScheduler,
            loss,
            global_step,
            scale):
        assert isinstance(loss, float)

        self.loss_metric.update(loss)
        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = (
                        self.frequent * self.batch_size /
                        (time.time() - self.tic)
                    )
                    self.tic = time.time()
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed = float("inf")
                    speed_total = float("inf")

                loss_str_format = f"{self.loss_metric.avg :.3f}"
                self.loss_metric.reset()

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.steps_total)
                time_for_end = time_total - time_now
                lr_1 = lr_scheduler.get_last_lr()[0]
                lr_2 = lr_scheduler.get_last_lr()[1]
                msg = f"rank:{int(speed) :d} "
                msg += f"total:{int(speed_total) :d} "
                msg += f"lr:[{lr_1 :.8f}][{lr_2 :.8f}] "
                msg += f"step:{global_step :d} "
                msg += f"amp:{int(scale) :d} "
                msg += f"required:{time_for_end :.1f} hours "
                msg += loss_str_format

                if self.rank == 0:
                    print(msg)
            else:
                self.init = True
                self.tic = time.time()


@torch.no_grad()
def euclidean_distance(x, y, topk=2):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(mat1=x, mat2=y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return torch.topk(dist, topk, largest=False)


@torch.no_grad()
def get_metric(
        query: torch.Tensor,
        query_label: list,
        gallery: torch.Tensor = None,
        gallery_label: list = None,
        l2norm=True,
        metric="rank1"):

    if gallery is None:
        query = query.cuda()
        if l2norm:
            query = normalize(query)
        if args.num_feat is not None:
            query = query[:, :args.num_feat]
        query_label = query_label
        list_pred = []
        num_feat = query.size(0)

        idx = 0
        is_end = 0
        while not is_end:
            if idx + 128 < num_feat:
                end = idx + 128
            else:
                end = num_feat
                is_end = 1

            _, index_pt = euclidean_distance(query[idx:end], query)
            index_np = index_pt.cpu().numpy()[:, 1]
            list_pred.append(index_np)
            idx += 128
        query_label = np.array(query_label).reshape(num_feat)
        pred = np.concatenate(list_pred).reshape(num_feat)
        rank_1 = np.sum(query_label == query_label[pred]) / num_feat
        rank_1 = float(rank_1)
        return rank_1 * 100
    else:
        query = query.cuda()
        query_label = query_label
        gallery = gallery.cuda()
        gallery_label = np.array(gallery_label)
        list_pred = []
        if l2norm:
            query = normalize(query)
            gallery = normalize(gallery)
        if args.num_feat is not None:
            query = query[:, :args.num_feat]
            gallery = gallery[:, :args.num_feat]
        num_feat = query.size(0)

        idx = 0
        is_end = 0
        while not is_end:
            if idx + 128 < num_feat:
                end = idx + 128
            else:
                end = num_feat
                is_end = 1

            _, index_pt = euclidean_distance(query[idx:end], gallery)
            index_np = index_pt.cpu().numpy()[:, 0]

            list_pred.append(index_np)
            idx += 128
        query_label = np.array(query_label).reshape(num_feat)
        pred = np.concatenate(list_pred).reshape(num_feat)
        rank_1 = np.sum(query_label == gallery_label[pred]) / num_feat
        rank_1 = float(rank_1)
        return rank_1 * 100


def get_transform(
        image_size: int = 224,
        is_train: bool = True
):
    from timm.data import create_transform
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=image_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if image_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(image_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=PIL.Image.BICUBIC))
    t.append(transforms.CenterCrop(image_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def sync_random_seed(seed=None, device="cuda"):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    distributed.broadcast(random_num, src=0)

    return random_num.item()


class DistributedSampler(_DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()
        # add extra samples to make it evenly divisible
        # in case that indices is shorter than half of total_size
        indices = (indices * math.ceil(self.total_size / len(indices)))[
            : self.total_size
        ]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


@torch.no_grad()
def get_metric_google_landmark(
        x_query,
        y_query,
        x_gallery,
        y_gallery) -> str:
    x_query = x_query.cuda()
    x_gallery = x_gallery.cuda()

    num = x_query.size(0)
    index = torch.zeros(num, 100, dtype=torch.long, device=x_query.device)
    score = torch.zeros(num, 100, device=x_query.device)
    num_feat = num

    idx = 0
    is_end = 0
    while not is_end:
        if idx + 128 < num_feat:
            end = idx + 128
        else:
            end = num_feat
            is_end = 1
        bs_score = torch.einsum("ik, jk -> ij", x_query[idx:end], x_gallery)
        score_pt, index_pt = torch.topk(bs_score, k=100, dim=1)
        index[idx:end] = index_pt
        score[idx:end] = score_pt
        idx += 128

    predictions_val = {}
    retrieval_solution_val = {}
    for i, some_list in enumerate(index):
        list_predictions = []
        for i_predictions in index[i]:
            list_predictions.append(y_gallery[i_predictions])
        predictions_val[i] = list_predictions

    for i, some_list in enumerate(y_query):
        retrieval_solution_val[i] = some_list.split(" ")

    mAP_val = mean_average_precision(predictions_val, retrieval_solution_val)
    return mAP_val


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

    return x, y_np


@torch.no_grad()
def evaluation(model: torch.nn.Module,
               dataset_dict: Dict, batch_size: int, num_workers: int):

    if "index" in dataset_dict:
        val, val_label = extract_feat(
            model, dataset_dict["val"], batch_size, num_workers)
        test, test_label = extract_feat(
            model, dataset_dict["test"], batch_size, num_workers)
        index, index_label = extract_feat(
            model, dataset_dict["index"], batch_size, num_workers)
        metric_val = get_metric_google_landmark(
            val, val_label, index, index_label)
        metric_test = get_metric_google_landmark(
            test, test_label, index, index_label)
        return metric_test, metric_val

    elif "test" in dataset_dict:
        dataset = dataset_dict["test"]
        x, y = extract_feat(model, dataset, batch_size, num_workers)
        metric = get_metric(x, y)
        return metric

    elif "query" in dataset_dict and "gallery" in dataset_dict:
        dataset_q = dataset_dict["query"]
        dataset_g = dataset_dict["gallery"]
        q, q_label = extract_feat(model, dataset_q, batch_size, num_workers)
        g, g_label = extract_feat(model, dataset_g, batch_size, num_workers)
        metric = get_metric(query=q, query_label=q_label,
                            gallery=g, gallery_label=g_label)
        return metric


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mean_average_precision(predictions, retrieval_solution, max_predictions=100):
    """Computes mean average precision for retrieval prediction.
  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account. For the Google Landmark Retrieval challenge, this should be set
      to 100.
  Returns:
    mean_ap: Mean average precision score (float).
  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
    # Compute number of test images.
    num_test_images = len(retrieval_solution.keys())

    # Loop over predictions for each query and compute mAP.
    mean_ap = 0.0
    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError(
                'Test image %s is not part of retrieval_solution' % key)

        # Loop over predicted images, keeping track of those which were already
        # used (duplicates are skipped).
        ap = 0.0
        already_predicted = set()
        num_expected_retrieved = min(
            len(retrieval_solution[key]), max_predictions)
        num_correct = 0
        for i in range(min(len(prediction), max_predictions)):
            if prediction[i] not in already_predicted:
                if prediction[i] in retrieval_solution[key]:
                    num_correct += 1
                    ap += num_correct / (i + 1)
                already_predicted.add(prediction[i])

        ap /= num_expected_retrieved
        mean_ap += ap

    mean_ap /= num_test_images

    return mean_ap


if __name__ == "__main__":
    main()
