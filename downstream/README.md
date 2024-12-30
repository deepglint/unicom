## RefCOCO Segmentation Evaluation: 

| Dataset     | Split   | MLCD-seg-7B | EVF-SAM | GLaMM | VisionLLM v2| LISA |
| :--         | :-:     | :-:  | :-:  | :-:  | :-:  | :-:  |
| RefCOCO     | val     | **83.6** | 82.4 | 79.5 | 79.2 | 74.9 |
| RefCOCO     | testA   | **85.3** | 84.2 | 83.2 | 82.3 | 79.1 |
| RefCOCO     | testB   | **81.5** | 80.2 | 76.9 | 77.0 | 72.3 |
| RefCOCO+    | val     | **79.4** | 76.5 | 72.6 | 68.9 | 65.1 |
| RefCOCO+    | testA   | **82.9** | 80.0 | 78.7 | 75.8 | 70.8 |
| RefCOCO+    | testB   | **75.6** | 71.9 | 64.6 | 61.8 | 58.1 |
| RefCOCOg    | val     | **79.9** | 78.2 | 74.2 | 73.3 | 67.9 |
| RefCOCOg    | test    | **80.5** | 78.3 | 74.9 | 74.8 | 70.6 |

---
## Evaluation  
Install the evaluation tool and execute the evaluation script:
```bash
bash ./eval/scripts/eval_refcoco.sh
```
---