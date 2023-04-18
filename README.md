# UNICOM  

[[paper]](https://arxiv.org/abs/2304.05884) [[gdrive]](https://drive.google.com/drive/folders/18wsNgZeNpjKAcIrWoffJ8o9UqmMHUBqN?usp=share_link)

For image representation:
1. ImageNet pretraining is not universal enough to generalize to diverse open-world objects.
2. Supervised learning is not scalable because manual annotation of large-scale training data is time-consuming, costly, and even infeasible.
3. Instance discrimination method (e.g., CLIP) can hardly encode the semantic structure of training data, because instance-wise contrastive learning always treats two samples as a negative pair, regardless of their semantic similarity.

UNICOM demonstrates superior performance in image retrieval, thanks to its ability to cluster **400000000** images into **1000000** pseudo classes using joint textual and visual features extracted by the CLIP model. Additionally, our use of a margin-based softmax loss (ArcFace) and random partial class/feature (PartialFC) selections enhances the robustness and compactness of the feature embedding. Our method outperforms state-of-the-art unsupervised and supervised image retrieval approaches, making it a powerful tool for researchers and practitioners in the field.

The model unicom was pre-trained on [laion400M](https://laion.ai/blog/laion-400-open-dataset/), and in the future, we will release the model trained on laion2B.


## Usage
First, install PyTorch 1.12 (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package.
On a CUDA GPU machine, the following will do the trick:

```shell
pip install torch torchvision
pip install tqdm timm
pip install git+https://github.com/deepglint/unicom.git
```

### API

The unicom module provides the following methods:

#### `unicom.available_models()`

Returns the names of the available unicom models.

#### `unicom.load(name)`

Returns the model and the TorchVision transform needed by the model, specified by the model name returned by `unicom.available_models()`. It will download the model as necessary.

## Result

### Supervised Image Retrieval

| Dataset     | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px |
|-------------|----------|----------|----------|----------------|
| SOP         | 87.1     | 88.8     | 89.9     | 91.2           |
| In-Shop     | 94.8     | 95.5     | 96.0     | 96.7           |
| INaturalist | 72.8     | 82.5     | 85.4     | 88.9           |

### Zero-Shot Image Retrieval

| Dataset     | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px |
|-------------|----------|----------|----------|----------------|
| CUB         | 83.7     | 86.5     | 88.5     | 89.2           |
| Cars        | 95.9     | 96.8     | 96.9     | 97.3           |
| SOP         | 70.0     | 70.4     | 72.7     | 74.5           |
| In-Shop     | 72.8     | 74.6     | 83.6     | 86.7           |
| INaturalist | 64.6     | 73.6     | 77.1     | 81.0           |


### Transfer-Learning on ImageNet1K

| Dataset    | ViT-B/32@384px | ViT-B/16@384px | ViT-L/14@518px |
|------------|----------------|----------------|----------------|
| ImageNet1k | 83.6           | 85.9           | 88.3           |


### Image Retrieval Eval
Zero-Shot CUB Dataset with a Single GPU.  

```shell
torchrun retrieval.py --eval --dataset cub --model_name ViT-B/32
```

Zero-Shot CUB Dataset with 8 GPUs.

```shell
torchrun --nproc_per_node 8 retrieval.py --eval --dataset cub --model_name ViT-B/32
```

### Image Retrieval Finetune
```shell
usage: retrieval.py [-h] [--batch_size BATCH_SIZE] [--dataset DATASET] [--debug DEBUG] [--epochs EPOCHS] [--eval] [--lr LR] [--lr_pfc_weight LR_PFC_WEIGHT] [--input_size INPUT_SIZE] [--gradient_acc GRADIENT_ACC] [--model_name MODEL_NAME]
                    [--margin_loss_m1 MARGIN_LOSS_M1] [--margin_loss_m2 MARGIN_LOSS_M2] [--margin_loss_m3 MARGIN_LOSS_M3] [--margin_loss_s MARGIN_LOSS_S] [--margin_loss_filter MARGIN_LOSS_FILTER] [--num_workers NUM_WORKERS] [--num_feat NUM_FEAT]
                    [--optimizer OPTIMIZER] [--output_dim OUTPUT_DIM] [--output OUTPUT] [--resume RESUME] [--sample_rate SAMPLE_RATE] [--seed SEED] [--transform TRANSFORM] [--weight_decay WEIGHT_DECAY] [--color_jitter COLOR_JITTER] [--aa AA] [--reprob REPROB]
                    [--remode REMODE] [--recount RECOUNT]

retrieval is a command-line tool that provides functionality for fine-tuning the Unicom model on retrieval tasks. With this tool, you can easily adjust the unicom model to achieve optimal performance on a variety of image retrieval tasks. Simply specify the task-
specific parameters and let the tool handle the rest.

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        The batch size to use for training and inference.
  --dataset DATASET     The dataset to load for training and evaluation.
  --debug DEBUG         A flag indicating whether to run the code in debug mode (with additional logging or other debugging aids).
  --epochs EPOCHS       The number of epochs to train the model for.
  --eval                A flag indicating whether to run model evaluation after training.
  --lr LR               The learning rate to use for training the model.
  --lr_pfc_weight LR_PFC_WEIGHT
                        The weight to apply to the learning rate for the Partial FC layer during training. Sure, when fine-tuning a pre-trained neural network, it is usually recommended to adjust the learning rates of different layers in order to achieve better
                        performance. For example, the learning rate of the backbone layers (i.e., the pre-trained layers) should be set lower because they already have learned features, while the learning rate of the Partial FC layer should be set higher, as it
                        needs to adapt to the new task.
  --input_size INPUT_SIZE
                        The size of the input images for the model.
  --gradient_acc GRADIENT_ACC
                        The number of times gradients are accumulated before updating the model's parameters.
  --model_name MODEL_NAME
                        The name of the pre-trained model to use for feature extraction.
  --margin_loss_m1 MARGIN_LOSS_M1
                        The margin parameter (m1) for the margin loss function.
  --margin_loss_m2 MARGIN_LOSS_M2
                        The margin parameter (m1) for the margin loss function.
  --margin_loss_m3 MARGIN_LOSS_M3
                        The margin parameter (m3) for the margin loss function.
  --margin_loss_s MARGIN_LOSS_S
                        The scale parameter (s) for the margin loss function.
  --margin_loss_filter MARGIN_LOSS_FILTER
                        The filter parameter for the margin loss function.
  --num_workers NUM_WORKERS
                        The number of workers to use for data loading.
  --num_feat NUM_FEAT   This parameter is used to set the dimensionality of the features sampled for use in model training and evaluation.
  --optimizer OPTIMIZER
                        The optimizer to use for the training process, default is AdamW.
  --output_dim OUTPUT_DIM
                        The desired dimensionality of the output embeddings in ViT.
  --output OUTPUT
  --resume RESUME       The path to a saved checkpoint to resume training from.
  --sample_rate SAMPLE_RATE
                        The negative sample rate to be used for partial FC. It helps to reduce memory usage, increase training speed And can significantly improve performance on datasets with high levels of noise
  --seed SEED           The random seed to use for reproducibility.
  --transform TRANSFORM
                        Transofrm in pytorch dataloader.
  --weight_decay WEIGHT_DECAY
                        Weight Decay.
  --color_jitter COLOR_JITTER
                        The amount of color jittering to apply during data augmentation.
  --aa AA               The amount of color jittering to apply during data augmentation. The default value is 'rand-m9-mstd0.5-inc1'.
  --reprob REPROB       The probability of replacing pixels during training using CutOut.
  --remode REMODE       The mode of replacement to use during training when using CutOut.
  --recount RECOUNT
```

## Citation

```latex
@inproceedings{anxiang_2023_unicom,
  title={Unicom: Universal and Compact Representation Learning for Image Retrieval},
  author={An, Xiang and Deng, Jiankang and Yang, Kaicheng and Li, Jiawei and Feng, Ziyong and Guo, Jia and Yang, Jing and Liu, Tongliang},
  booktitle={ICLR},
  year={2023}
}
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  pages={4690--4699},
  year={2019}
}
@inproceedings{anxiang_2022_partialfc,
    author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
    title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
    booktitle={CVPR},
    year={2022},
    pages={4042-4051}
}
```
