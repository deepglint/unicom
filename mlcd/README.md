[![Arxiv](https://img.shields.io/badge/arXiv-2407.17331-red)](https://arxiv.org/abs/2407.17331) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/self-supervised-image-classification-on)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=multi-label-cluster-discrimination-for-visual)


### Performance


#### A. MLLMs Evaluation Results
To evaluate MLCD’s performance within multimodal large language models (MLLMs), we replaced the CLIP model in LLaVA-NeXT with the MLCD model. We paired this with the Qwen2.5-7B language model. For reproducibility, we utilized the LLaVA-Pretrain dataset for pre-training and the LLaVA-NeXT-Data for structured fine-tuning. The evaluation results confirm that the MLCD model performs exceptionally well across multiple benchmarks, underscoring its effectiveness in MLLMs.


| Vision Tower                                                                                  | RoPE2D | ChartQA   | DocVQA    | InfoVQA   | OCRBench   | MMMU      |
| :-------------------------------------------------------------------------------------------- | :----: | :-------- | :-------- | :-------- | :--------- | :-------- |
| CLIP (ViT-L-14-336px)                                                                         |   ×    | 66.52     | 75.21     | 38.88     | 525.00     | 44.20     |
| SigLIP (ViT-SO400M-384px)                                                                     |   ×    | 69.28     | 76.71     | 41.38     | 554.00     | 46.78     |
| DFN5B (ViT-H-14-378px)                                                                        |   ×    | 64.36     | 70.87     | 38.59     | 473.00     | **48.00** |
| **[HF:MLCD (ViT-L-14-336px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336)**   |   ×    | 67.84     | 76.46     | 43.48     | 531.00     | 44.30     |
| **[HF:MLCD (ViT-bigG-14-336px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-336)** |   √    | 71.07     | 79.63     | 44.38     | 572.00     | 46.78     |
| **[HF:MLCD (ViT-bigG-14-448px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-448)** |   √    | **73.80** | **83.34** | **46.59** | **582.00** | 46.00     |



#### B. Linear Probe Evaluation Results
This table presents the results of linear probe evaluations comparing CLIP and MLCD models on the ViT_L_14_336px architecture across various datasets. The linear probe test freezes the pre-trained model's weights and trains a linear classifier on top to assess how well the model's representations generalize to different tasks.


The results of the ImageNet linear probe are as follows:

| Model Name             | ImageNet Linear Probe | Hugging Face                                                                               |
| :--------------------- | :-------------------: | :----------------------------------------------------------------------------------------- |
| MLCD-ViT-B-32-224px    |         79.1          | [HF:MLCD-ViT-B-32-224px](https://huggingface.co/DeepGlint-AI/mlcd-vit-base-patch32-224)    |
| MLCD-ViT-L-14-336px    |         86.3          | [HF:MLCD-ViT-L-14-336px](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336)   |
| MLCD-ViT-bigG-14-224px |         87.1          | [HF:MLCD-ViT-bigG-14-224px](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-224) |


| Dataset                      | MLCD (ViT_L_14_336px) | CLIP (ViT_L_14_336px) |
| :--------------------------- | :-------------------- | :-------------------- |
| Food101                      | **96.21**             | 95.90                 |
| CIFAR-10                     | **99.36**             | 97.90                 |
| CIFAR-100                    | **93.69**             | 87.40                 |
| Birdsnap                     | **88.18**             | 79.90                 |
| SUN397                       | **87.96**             | 82.20                 |
| Stanford Cars                | **95.16**             | 91.50                 |
| FGVC Aircraft                | **86.38**             | 71.60                 |
| Describable Textures Dataset | **86.70**             | 83.00                 |
| Oxford-IIIT Pets             | **96.27**             | 95.10                 |
| Caltech-101                  | **97.92**             | 96.00                 |
| Flowers102                   | **99.58**             | 99.20                 |
| ImageNet                     | **86.10**             | 85.40                 |


### convert pytorch2huggingface

```python3

python convert_vit_bigG_14_rope2d_to_hf.py \
--pytorch_dump_folder_path mlcd-vit-bigG-patch14-336 \
--checkpoint_path MLCD_ViT_bigG_14_336px_pytorch.pt \
--image_size 336
```
