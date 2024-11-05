<p align="center" width="100%">
<img src="asserts/logo.png" alt="/80dafc65-cda6-4001-aecf-3989ea9d2f7c.webp" width=30%>
</p>
<div>


# UNICOM & MLCD

This repository is dedicated to building foundational visual models using large-scale datasets such as LAION400M and COYO700M. We employ sample-to-cluster contrastive learning to optimize performance. Our models have been thoroughly validated across various tasks, including multimodal visual large language models (e.g., LLaVA), image retrieval, and image classification.


# Latest News
🤖 [2024/11] We have launched the [MLCD-Embodied-7B](#mlcd-embodied), which can reach the level of GPT-4V in embodied capabilities and possesses excellent general understanding abilities. For more details, please click &rarr; [MLCD-Embodied.md](docs/MLCD_Embodied.md).  
🤗 [2024/10] We release [MLCD-NeXT-7B](https://huggingface.co/DeepGlint-AI/llava-mlcd-qwen2.5-7b) to Hugging Face.  
🤌 [2024/07] [MLCD](#multi-label-cluster-discrimination-mlcd) was accepted to ECCV2024.  
🌍 [2023/03] [UNICOM](#unicom) was accepted to ICLR2023.  

---

## MLCD-Embodied
<a name="mlcd-embodied"></a>
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/DeepGlint-AI/MLCD-Embodied-7B)  
More details about MLCD-Embodied can be found in the [MLCD-Embodied.md](docs/MLCD_Embodied.md) file.  

Some test results are as follows:
<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
  <img src="asserts/radar_openeqa.png" alt="Image 1" style="width: 45%;">
  <img src="asserts/radar_robomembar.png" alt="Image 2" style="width: 45%;">
</div>







## Multi-Label Cluster Discrimination (MLCD)
<a name="multi-label-cluster-discrimination-mlcd"></a>
[![Arxiv](https://img.shields.io/badge/arXiv-2407.17331-red)](https://arxiv.org/abs/2407.17331) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336)

<p align="left" width="100%">
<img src="https://github.com/user-attachments/assets/d037ef08-a72f-421a-bdb8-d9b187794989" width="80%">


While CLIP models have shown excellence in many tasks via image-text contrastive learning, they often struggle with encoding complex semantic structures within images. To address this limitation, we introduce **Multi-Label Cluster Discrimination (MLCD)**.

MLCD improves upon traditional approaches by clustering the the LAION dataset, which contains billions of images, into one million centers and assigning multiple closest clusters as labels to each image. This technique accounts for the presence of multiple objects within a single image. We also introduce a novel multi-label classification loss, which separately handles positive and negative class losses, minimizing label ambiguity. Our experiments demonstrate that MLCD achieves state-of-the-art performance in linear probe. Moreover, MLCD shows significant potential when integrated with multimodal large language models.

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
### Usage
#### A. Installation

##### **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone https://github.com/deepglint/unicom
cd unicom

# Upgrade pip and install necessary dependencies
pip install --upgrade pip
pip install -e ".[train]"
```

#### B. Training

**Stage 1: Pretraining**
```bash
bash scripts/pretrain_mlcd.sh
```

**Stage 2: Instructional Finetuning**
```bash
bash scripts/finetune_mlcd.sh
```


#### C. Evaluation  
Install the evaluation tool and execute the evaluation script:
```bash
pip install lmms-eval==0.2.0
bash eval.sh
```
---

## UNICOM
<a name="unicom"></a>
[![Arxiv](https://img.shields.io/badge/arXiv-2304.05884-red)](https://arxiv.org/abs/2304.05884) [![Google Drive](https://img.shields.io/badge/Google%20Drive-Model-yellow)](https://drive.google.com/drive/folders/18wsNgZeNpjKAcIrWoffJ8o9UqmMHUBqN?usp=share_link)


For image representation:
1. ImageNet pretraining is not universal enough to generalize to diverse open-world objects.
2. Supervised learning is not scalable because manual annotation of large-scale training data is time-consuming, costly, and even infeasible.
3. Instance discrimination method (e.g., CLIP) can hardly encode the semantic structure of training data, because instance-wise contrastive learning always treats two samples as a negative pair, regardless of their semantic similarity.

UNICOM demonstrates superior performance in image retrieval, thanks to its ability to cluster **400000000** images into **1000000** pseudo classes using joint textual and visual features extracted by the CLIP model. Additionally, our use of a margin-based softmax loss (ArcFace) and random partial class/feature (PartialFC) selections enhances the robustness and compactness of the feature embedding. Our method outperforms state-of-the-art unsupervised and supervised image retrieval approaches, making it a powerful tool for researchers and practitioners in the field.

### Usage

For detailed instructions, please refer to the UNICOM  [Documentation](docs/UNICOM.md).


## Contributors
Thanks so much to all of our amazing contributors!

<a href="https://github.com/deepglint/unicom/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=deepglint/unicom&r=&timestamp=<?=time()?>"/>
</a>


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

We extend our deepest gratitude to the creators and contributors of the following projects:  
1. [llava-next](https://github.com/LLaVA-VL/LLaVA-NeXT): The comprehensive codebase for training Vision-Language Models (VLMs).  
2. [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): The robust tool for evaluating Vision-Language Models (VLMs).
3. [OpenEQA](https://github.com/facebookresearch/open-eqa): A wonderful benchmark for Embodied Question Answering.
4. [RoboVQA](https://github.com/google-deepmind/robovqa): Provide high level reasoning model and dataset for robotics.

Their exceptional work has been instrumental to our research and development efforts.





