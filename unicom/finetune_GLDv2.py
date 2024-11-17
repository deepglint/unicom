import argparse
import math
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
from dataset.google_landmark import (get_testset_gallery,
                                     get_testset_query_test,
                                     get_testset_query_val)
from dataset.google_landmark_dali import get_loader_train
from partial_fc import CombinedMarginLoss, PartialFC_V2
from scipy import ndimage
from torch import distributed, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Subset

import unicom

parser = argparse.ArgumentParser(
    description="retrieval is a command-line tool that provides functionality for fine-tuning the Unicom model on retrieval tasks. With this tool, you can easily adjust the unicom model to achieve optimal performance on a variety of image retrieval tasks. Simply specify the task-specific parameters and let the tool handle the rest.")
parser.add_argument("--batch_size", default=16, type=int, help="The batch size to use for training and inference.")
parser.add_argument("--epochs", type=int, default=16, help="The number of epochs to train the model for.")
parser.add_argument("--eval", action="store_true", help="Whether to only evaluate the model.")
parser.add_argument("--lr", type=float, default=0.0001, help="The learning rate to use for training the model.")
parser.add_argument("--lr_pfc_weight", type=float, default=3.0, help="The weight to apply to the learning rate for the Partial FC layer during training. Sure, when fine-tuning a pre-trained neural network, it is usually recommended to adjust the learning rates of different layers in order to achieve better performance. For example, the learning rate of the backbone layers (i.e., the pre-trained layers) should be set lower because they already have learned features, while the learning rate of the Partial FC layer should be set higher, as it needs to adapt to the new task.")
parser.add_argument("--input_size", default=512, type=int, help="The size of the input images for the model.")
parser.add_argument("--gradient_acc", default=1, type=int, help="The number of times gradients are accumulated before updating the model's parameters.")
parser.add_argument("--model_name", default="ViT-L/14", help="The name of the pre-trained model to use for feature extraction.")
parser.add_argument("--margin_loss_m1", type=float, default=1.0, help="The margin parameter (m1) for the margin loss function.")
parser.add_argument("--margin_loss_m2", type=float, default=0.5, help="The margin parameter (m1) for the margin loss function.")
parser.add_argument("--margin_loss_m3", type=float, default=0.0, help="The margin parameter (m3) for the margin loss function.")
parser.add_argument("--margin_loss_s", type=float, default=32, help="The scale parameter (s) for the margin loss function.")
parser.add_argument("--margin_loss_filter", type=float, default=0.0, help="The filter parameter for the margin loss function.")
parser.add_argument("--num_workers", default=16, type=int, help="The number of workers to use for data loading.")
parser.add_argument("--num_feat", default=None, type=int, help="This parameter is used to set the dimensionality of the features sampled for use in model training and evaluation. ")
parser.add_argument("--output_dim", type=int, default=768, help="The desired dimensionality of the output embeddings in ViT.")
parser.add_argument("--output", default="checkpoints", help="")
parser.add_argument("--sample_rate", default=0.5, type=float, help="The negative sample rate to be used for partial FC. It helps to reduce memory usage, increase training speed And can significantly improve performance on datasets with high levels of noise")
parser.add_argument("--seed", type=int, default=1024, help="The random seed to use for reproducibility.")
parser.add_argument("--transform", default=None, type=str, help="Transofrm in pytorch dataloader.")
parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay.")
parser.add_argument("--num_classes", type=int, default=81314, help="")


args = parser.parse_args()

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)



def zoom_state_dict(state_dict, input_size):
    patch_size = state_dict['patch_embed.proj.weight'].size(2)
    pos_embed = state_dict.pop("pos_embed").float().cpu().numpy()
    gs_old = int(np.sqrt(len(pos_embed[0])))
    gs_new = input_size // patch_size
    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = ndimage.zoom(pos_embed[0].reshape(gs_old, gs_old, -1), zoom, order=1)
    state_dict["pos_embed"] = torch.from_numpy(posemb_grid.reshape(1, gs_new * gs_new, -1)).to(state_dict["patch_embed.proj.weight"].device)
    weight = state_dict.pop("feature.0.weight").float().cpu().numpy()
    weight = weight.reshape(gs_old, gs_old, weight.shape[0], -1)
    zoom_weight = (gs_new / gs_old, gs_new / gs_old, 1, 1)
    weight_zoomed = ndimage.zoom(weight, zoom_weight, order=1).reshape(-1, weight.shape[2])
    state_dict["feature.0.weight"] = torch.from_numpy(weight_zoomed.transpose(1, 0)).to(state_dict["patch_embed.proj.weight"].device)
    return state_dict

def main():

    distributed.init_process_group(backend="nccl")

    if rank == 0:
        for arg in vars(args):
            print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
    os.makedirs(args.output, exist_ok=True)

    if args.eval:
        model, transform_clip = unicom.load(args.model_name)
    else:
        if args.model_name == "ViT-B/16":
            from unicom.vision_transformer import VisionTransformer, _transform
            model = VisionTransformer(
                input_size=512, patch_size=16, in_channels=3, dim=768, embedding_size=args.output_dim,
                depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=False)
            transform_clip = _transform(512)
            state_dict = torch.load("/root/.cache/unicom/FP16-ViT-B-32.pt", "cpu")
            state_dict = zoom_state_dict(state_dict, 512)
            model.load_state_dict(state_dict)
        elif args.model_name == "ViT-B/32":
            from unicom.vision_transformer import VisionTransformer, _transform
            model = VisionTransformer(
                input_size=512, patch_size=32, in_channels=3, dim=768, embedding_size=args.output_dim,
                depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=False)
            transform_clip = _transform(512)
            state_dict = torch.load("/root/.cache/unicom/FP16-ViT-B-16.pt", "cpu")
            state_dict = zoom_state_dict(state_dict, 512)
            model.load_state_dict(state_dict)
        elif args.model_name == "ViT-L/14":
            from unicom.vision_transformer import VisionTransformer, _transform
            model = VisionTransformer(
                input_size=512, patch_size=14, in_channels=3, dim=1024, embedding_size=args.output_dim,
                depth=24, num_heads=16, drop_path_rate=0.1, using_checkpoint=False)
            transform_clip = _transform(512)
            state_dict = torch.load("/root/.cache/unicom/FP16-ViT-L-14-336px.pt", "cpu")
            state_dict = zoom_state_dict(state_dict, 512)
            model.load_state_dict(state_dict)

        else:
            raise

    dataset_query_val = get_testset_query_val(transform_clip)
    dataset_query_test = get_testset_query_test(transform_clip)
    dataset_gallery = get_testset_gallery(transform_clip)
    dataset_dict = {
        "val": dataset_query_val,
        "test": dataset_query_test,
        "index": dataset_gallery
    }

    model.train()
    model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.compile(model)


    if args.eval:
        score_test, score_val = evaluation(
            model, dataset_dict,
            args.batch_size,
            num_workers=args.num_workers)
        print(f"Test: {score_test:.3f} Val: {score_val:.3f}")
        return
    else:
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

        module_partial_fc = PartialFC_V2(
            margin_loss, args.output_dim, args.num_classes,
            args.sample_rate, False, sample_num_feat=None)
        module_partial_fc.train().cuda()

        opt = torch.optim.AdamW(
            params=[
                {"params": backbone.parameters()},
                {"params": module_partial_fc.parameters(), "lr": args.lr * args.lr_pfc_weight}],
            lr=args.lr, weight_decay=args.weight_decay)

        num_train_set = 1580470
        loader_train = get_loader_train(
            args.batch_size, args.input_size, args.input_size * 256 / 224, 4)

        steps_per_epoch = num_train_set // world_size // args.batch_size + 1
        total_steps = args.epochs * steps_per_epoch

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

        callback_func = SpeedCallBack(10, total_steps, args.batch_size)
        auto_scaler = torch.cuda.amp.grad_scaler.GradScaler(
            growth_interval=200)
        global_step = 0

        for epoch in range(0, args.epochs):
            backbone.train()
            for _, (img, local_labels) in enumerate(loader_train):
                img = img.cuda()
                local_labels = local_labels.long().cuda()
                with torch.cuda.amp.autocast(True):
                    local_embeddings = backbone(img)
                local_embeddings = local_embeddings.float()

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
            if hasattr(loader_train, "reset"):
                loader_train.reset()
            score_test, score_val = evaluation(
                model, dataset_dict,
                args.batch_size,
                num_workers=args.num_workers)
            model_name = args.model_name.replace("/", "_").replace("-", "_")

            if rank == 0:
                print(f"Test: {score_test:.3f} Val: {score_val:.3f}, epoch is {epoch}")
                torch.save(model.state_dict(), f"{args.output}/{model_name}_{epoch}_test_{score_test:.3f}_val_{score_val:.3f}.pth")
            torch.save(module_partial_fc.state_dict(), f"{args.output}/{model_name}_pfc_{epoch}_{rank:02}.pt")
            model.train()



class SpeedCallBack(object):
    def __init__(self, frequent, total_steps, batch_size):
        self.batch_size = batch_size
        self.frequent = frequent
        self.total_steps = total_steps
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
                time_total = time_now / ((global_step + 1) / self.total_steps)
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

    else:
        raise

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
