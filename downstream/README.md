[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcocog)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcocog?p=multi-label-cluster-discrimination-for-visual)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-5)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-5?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-3)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-3?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcocog-1)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcocog-1?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-8)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-8?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-4)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-4?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco-9)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-9?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco?p=multi-label-cluster-discrimination-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-cluster-discrimination-for-visual/referring-expression-segmentation-on-refcoco)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco?p=multi-label-cluster-discrimination-for-visual) 


# MLCD-Seg
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-MLCD_SEG_Model-yellow)](https://huggingface.co/DeepGlint-AI/MLCD-Seg-7B)

This repository is dedicated to researching the application of multimodal large models in downstream tasks through an end-to-end approach. At present, the segmentation part has achieved excellent results in the reference segmentation project


## RefCOCO Segmentation Evaluation: 

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

---
## Evaluation  
Install the evaluation tool and execute the evaluation script:
```bash
bash ./eval/scripts/eval_refcoco.sh
```
---

## Citations
```
@misc{mlcdseg_wukun,
  author = {Wu, Kun and Xie, Yin and Zhou, Xinyu and An, Xiang, and Deng, Jiankang},
  title = {MLCD-seg-7B},
  year = {2024},
  url = {https://github.com/deepglint/unicom/tree/main/downstream},
}
```
