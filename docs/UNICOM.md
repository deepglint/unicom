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

## Results and Evaluation

### Result Transfer-Learning on ImageNet1K

| Dataset    | ViT-B/32@384px | ViT-B/16@384px | ViT-L/14@518px |
| ---------- | -------------- | -------------- | -------------- |
| ImageNet1k | 83.6           | 85.9           | 88.3           |

### Result KNN on ImageNet1K
| Dataset    | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px |
| ---------- | -------- | -------- | -------- | -------------- |
| ImageNet1K | 74.5     | 78.8     | 81.2     | 81.6           |


### Result of Supervised Image Retrieval

| Dataset     | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px |
| ----------- | -------- | -------- | -------- | -------------- |
| SOP         | 87.1     | 88.8     | 89.9     | 91.2           |
| In-Shop     | 94.8     | 95.5     | 96.0     | 96.7           |
| INaturalist | 72.8     | 82.5     | 85.4     | 88.9           |

### Result of Zero-Shot Image Retrieval

| Dataset     | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px |
| ----------- | -------- | -------- | -------- | -------------- |
| CUB         | 83.7     | 86.5     | 88.5     | 89.2           |
| Cars        | 95.9     | 96.8     | 96.9     | 97.3           |
| SOP         | 70.0     | 70.4     | 72.7     | 74.5           |
| In-Shop     | 72.8     | 74.6     | 83.6     | 86.7           |
| INaturalist | 64.6     | 73.6     | 77.1     | 81.0           |


### Eval Image Retrieval
Zero-Shot CUB Dataset with a Single GPU.  

```shell
torchrun retrieval.py --eval --dataset cub --model_name ViT-B/32
```

Zero-Shot CUB Dataset with 8 GPUs.

```shell
torchrun --nproc_per_node 8 retrieval.py --eval --dataset cub --model_name ViT-B/32
```

### Eval KNN
```shell  

torchrun --nproc_per_node 8 knn.py --train-dataset /imagenet/train/ --val-dataset /imagenet/val/ --num-workers 4 --model-name ViT-B/32
```  

## Vis ZeroShot Retrieval

#### 1. **Food-101**
![image](../examples/vis_food101.jpg)
#### 2. **Describable Textures Dataset**
![image](../examples/vis_dtd.jpg)
