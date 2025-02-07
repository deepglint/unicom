[![Arxiv](https://img.shields.io/badge/arXiv-2407.17331-red)](https://arxiv.org/abs/2407.17331) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/self-supervised-image-classification-on)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=multi-label-cluster-discrimination-for-visual)


### Evaluation

#### A. MLLMs Evaluation Results
To evaluate MLCD’s performance within multimodal large language models (MLLMs), we replaced the CLIP model in LLaVA-NeXT with the MLCD model. We paired this with the Qwen2.5-7B language model. For reproducibility, we utilized the LLaVA-Pretrain dataset for pre-training and the LLaVA-NeXT-Data for structured fine-tuning. The evaluation results confirm that the MLCD model performs exceptionally well across multiple benchmarks, underscoring its effectiveness in MLLMs.


| Vision Tower    | [MLCD (ViT_L_14_336px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336) | CLIP (ViT_L_14_336px) |
|:----------------|:-------------|:-------------|
| LLM             | Qwen2.5-7B   |   Qwen2.5-7B |
| AI2D            | **76.98**    | 73.15        |
| GQA             | **64.17**    | 63.31        |
| ScienceQA-Img   | **78.09**    | 76.35        |
| InfoVQA-Val     | **43.48**    | 38.88        |
| MMBenchCN-Dev  | **74.83**    | 72.51        |
| MMBenchEN-Dev  | **76.37**    | 74.57        |
| SeedBench       | **68.20**    | 66.80        |
| SeedBench-Img   | **73.75**    | 72.72        |
| MMStar          | **50.98**    | 48.98        |
| MMMU            | **44.30**    | 44.20        |
| POPE            | 88.69        | **88.83**    |
| ChartQA         | **67.84**    | 66.52        |
| DocVQA-Val      | **76.46**    | 75.21        |
| TextVQA-Val     | 61.69        | **62.47**    |
| OCRBench        | **531**      | 525       |
| MME(cognition)  | **432**      | 384          |
| MME(perception) | **1598**     | 1512         |




#### B. Linear Probe Evaluation Results
This table presents the results of linear probe evaluations comparing CLIP and MLCD models on the ViT_L_14_336px architecture across various datasets. The linear probe test freezes the pre-trained model's weights and trains a linear classifier on top to assess how well the model's representations generalize to different tasks.

| Dataset        | MLCD (ViT_L_14_336px) | CLIP (ViT_L_14_336px) |
|:---------------|:----------------------|:----------------------|
| Food101        | **96.21**             | 95.90                 |
| CIFAR-10       | **99.36**             | 97.90                 |
| CIFAR-100      | **93.69**             | 87.40                 |
| Birdsnap       | **88.18**             | 79.90                 |
| SUN397         | **87.96**             | 82.20                 |
| Stanford Cars  | **95.16**             | 91.50                 |
| FGVC Aircraft  | **86.38**             | 71.60                 |
| Describable Textures Dataset | **86.70** | 83.00                 |
| Oxford-IIIT Pets | **96.27**          | 95.10                 |
| Caltech-101    | **97.92**             | 96.00                 |
| Flowers102     | **99.58**             | 99.20                 |
| ImageNet       | **86.10**             | 85.40                 |



