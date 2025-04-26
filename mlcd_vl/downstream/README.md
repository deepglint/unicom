<a href="https://arxiv.org/pdf/2407.17331"><img src="https://img.shields.io/badge/arXiv-2407.17331-b31b1b" alt="arXiv"></a>
<a href='https://huggingface.co/DeepGlint-AI/MLCD-Seg'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-green'></a>
</div>



## Example:

![output](https://github.com/user-attachments/assets/85c023a1-3e0c-4ea5-a764-1eb9ee0fbddf)


## RefCOCO Segmentation Evaluation Results:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcocog)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcocog?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-5)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-5?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-3)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-3?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcocog-1)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcocog-1?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-8)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-8?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-4)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-4?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-9)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-9?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco?p=multi-label-cluster-discrimination-for-visual) 



| Dataset     | Split   | MLCD-seg-7B | EVF-SAM | GLaMM | VisionLLM v2| LISA |
| :--         | :-:     | :-:  | :-:  | :-:  | :-:  | :-:  |
| RefCOCO     | val     | **83.6** | 82.4 | 79.5 | 79.2 | 74.9 |
| RefCOCO     | testA   | **85.3** | 84.2 | 83.2 | 82.3 | 79.1 |
| RefCOCO     | testB   | **81.5** | 80.2 | 76.9 | 77.0 | 72.3 |
| RefCOCO+    | val     | **79.4** | 76.5 | 72.6 | 68.9 | 65.1 |
| RefCOCO+    | testA   | **82.9** | 80.0 | 78.7 | 75.8 | 70.8 |
| RefCOCO+    | testB   | **75.6** | 71.9 | 64.6 | 61.8 | 58.1 |
| RefCOCOg    | val     | **79.7** | 78.2 | 74.2 | 73.3 | 67.9 |
| RefCOCOg    | test    | **80.5** | 78.3 | 74.9 | 74.8 | 70.6 |

## How to use:

If you just want to use this code, please refer to this sample below
```python
model_path = "DeepGlint-AI/MLCD-Seg" # or use your local path
mlcd_seg = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# Assuming you have an image named test.jpg
seg_img = Image.open("asserts/example.jpg").convert('RGB')
seg_prompt = "Could you provide a segmentation mask for the right giraffe in this image?"
pred_mask = model.seg(seg_img, seg_prompt, tokenizer, force_seg=False)
```

If you want to use this code measurement dataset (e.g. refcoco), then you need to use the following method
```python
model_path = "DeepGlint-AI/MLCD-Seg" # or use your local path
mlcd_seg = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# Assuming you have an image named test.jpg
seg_img = Image.open("asserts/example.jpg").convert('RGB')
seg_prompt = "Could you provide a segmentation mask for the right giraffe in this image?"
pred_mask = model.seg(seg_img, seg_prompt, tokenizer, force_seg=True)
```

## Intstallation

```bash
# Create environment from file
conda create -n mlcd_seg python=3.10
conda activate mlcd_seg

pip install -r requirements.txt
```


## Docker
```bash
# PyTorch Docker

```bash
# Build the Docker image
docker build -t mlcd_seg .

# Run the Docker container with GPU support
docker run -it --rm --gpus all mlcd_seg bash
```


## Citations
```
@misc{mlcdseg_wukun,
  author = {Wu, Kun and Xie, Yin and Jie, Yu and Zhou, Xinyu and An, Xiang, Feng, Ziyong and Deng, Jiankang},
  title = {MLCD-Seg},
  year = {2025},
  url = {https://github.com/deepglint/MLCD_SEG},
}
@inproceedings{anxiang_2024_mlcd,
  title={Multi-label Cluster Discrimination for Visual Representation Learning},
  author={An, Xiang and Yang, Kaicheng and Dai, Xiangzi and Feng, Ziyong and Deng, Jiankang},
  booktitle={ECCV},
  year={2024}
}
```
