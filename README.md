# UNICOM & MLCD

This repository focuses on creating foundational visual models through large-scale data, such as LAION400M and COYO700M, by employing sample-to-cluster contrastive learning. The models have been validated across various tasks, including multimodal visual large language models (e.g., LLaVA), image retrieval, and image classification.



## MLCD

[[Paper]](https://arxiv.org/abs/2407.17331) [[Hugging Face]](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336)



CLIP excels in various tasks due to image-text contrastive learning but struggles with encoding semantic structures. We propose Multi-Label Cluster Discrimination (MLCD) to address this. MLCD clusters the LAION-400M dataset into one million centers, using multiple closest centers as labels to account for multiple objects in images. A novel multi-label classification loss separates positive and negative class losses, reducing ambiguity. Experiments show MLCD achieves state-of-the-art performance in linear probe, zero-shot classification, and image-text retrieval tasks. Additionally, MLCD demonstrates promising results in multimodal large language models.


### Evaluation

#### A. MLLMs Evaluation Results
In our experiments, we replaced the CLIP model in [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) with the MLCD model to demonstrate the performance of the MLCD model in Multimodal Large Language Models (MLLMs). For the language model, we used [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B). Additionally, to facilitate the reproducibility of results, we used the [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) for pre-training and the [LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) for structured fine-tuning. The evaluation results show that the modified model performs exceptionally well across multiple benchmarks, validating the effectiveness of the MLCD model within MLLMs.

| Vision Tower    | MLCD (ViT_L_14_336px) | CLIP (ViT_L_14_336px) |
|:----------------|:-------------|:-------------|
| LLM             | Qwen2.5-7B   |   Qwen2.5-7B |
| AI2D            | **76.98**    | 73.15        |
| ScienceQA_img   | **78.09**    | 76.35        |
| GQA             | **64.17**    | 63.31        |
| InfoVQA_val     | **43.48**    | 38.88        |
| MMBench_cn_dev  | **74.83**    | 72.51        |
| MMBench_en_dev  | **76.37**    | 74.57        |
| MME(cognition)  | **432**      | 384          |
| MME(perception) | **1598**     | 1512         |
| SeedBench       | **68.20**    | 66.80        |
| SeedBench_img   | **73.75**    | 72.72        |
| MMStar          | **50.98**    | 48.98        |
| MMMU            | **44.30**    | 44.20        |
| OCRBench        | **531.00**   | 525.00       |
| ChartQA         | **67.84**    | 66.52        |
| DocVQA_val      | **76.46**    | 75.21        |
| POPE            | 88.69        | **88.83**    |
| TextVQA_val     | 61.69        | **62.47**    |


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

### Usage
#### A. Installation

##### 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone https://github.com/deepglint/unicom
cd unicom
```

##### 2. **Install the inference package:**
```bash
pip install --upgrade pip
pip install -e ".[train]"
```

#### B. Training

```bash
# stage 1: pretrain
bash scripts/pretrain_mlcd.sh

# stage 2: instrcuted finetune
bash scripts/finetune_mlcd.sh
```


#### C. Evaluation
```bash
pip install lmms-eval==0.2.0
bash eval.sh
```


## UNICOM

[[Paper]](https://arxiv.org/abs/2304.05884) [[Google Drive]](https://drive.google.com/drive/folders/18wsNgZeNpjKAcIrWoffJ8o9UqmMHUBqN?usp=share_link)

For image representation:
1. ImageNet pretraining is not universal enough to generalize to diverse open-world objects.
2. Supervised learning is not scalable because manual annotation of large-scale training data is time-consuming, costly, and even infeasible.
3. Instance discrimination method (e.g., CLIP) can hardly encode the semantic structure of training data, because instance-wise contrastive learning always treats two samples as a negative pair, regardless of their semantic similarity.

UNICOM demonstrates superior performance in image retrieval, thanks to its ability to cluster **400000000** images into **1000000** pseudo classes using joint textual and visual features extracted by the CLIP model. Additionally, our use of a margin-based softmax loss (ArcFace) and random partial class/feature (PartialFC) selections enhances the robustness and compactness of the feature embedding. Our method outperforms state-of-the-art unsupervised and supervised image retrieval approaches, making it a powerful tool for researchers and practitioners in the field.

### Usage

For detailed instructions, please refer to the UNICOM  [Documentation](docs/UNICOM.md).


## Dataset Contributors
This project would not have been possible without the invaluable contributions of the following individuals, who have been instrumental in data scraping and collection:  
Thank you to all the contributors for their hard work and dedication!

| Contributor      | Emial    |
|------------------|----------|
| **Bin Qin**         | skyqin@gmail.com              |
| **Lan Wu**          | bah-wl@hotmail.com            |
| **Haiqiang Jiang**  | haiqiangjiang@deepglint.com   |
| **Yuling Wu**       | yulingwu@deepglint.com        |

## Citation

```latex
@inproceedings{anxiang_2024_mlcd,
  title={Multi-label Cluster Discrimination for Visual Representation Learning},
  author={An, Xiang and Yang, Kaicheng and Dai, Xiangzi and Feng, Ziyong and Deng, Jiankang},
  booktitle={ECCV},
  year={2024}
}
@inproceedings{anxiang_2023_unicom,
  title={Unicom: Universal and Compact Representation Learning for Image Retrieval},
  author={An, Xiang and Deng, Jiankang and Yang, Kaicheng and Li, Jiawei and Feng, Ziyong and Guo, Jia and Yang, Jing and Liu, Tongliang},
  booktitle={ICLR},
  year={2023}
}
@inproceedings{anxiang_2022_partialfc,
    author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
    title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
    booktitle={CVPR},
    year={2022},
}
@inproceedings{deng_2019_arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}
```

## Acknowledgement
[LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT): the codebase for training VLMs. Thanks for their wonderful work.
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): the tool for evaluating VLMs.
