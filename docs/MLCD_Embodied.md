# MLCD-Embodied


## Performance in RoboVQA and OpenEQA



|                |                   | MLCD-Embodied-7B | LLaVA OneVision-7B | GPT-4V | RoboMamba |
|----------------|-------------------|-------------------|--------------------|--------|-----------|
| **RoboVQA**   | BLEU1             | **73.16**        | 38.12             | -      | 54.9      |
|                | BLEU2             | **66.39**        | 33.56             | -      | 44.2      |
|                | BLEU3             | **60.61**        | 31.76             | -      | 39.5      |
|                | BLEU4             | **56.56**        | 30.97             | -      | 36.3      |
| **OpenEQA**    | OBJECT-STATE-RECOGNITION | **71.83** | -           | 63.2   | -         |
|                | OBJECT-RECOGNITION        | **49.46** | -           | 43.4   | -         |
|                | FUNCTIONAL-REASONING      | 54.38 | -           | **57.4** | -       |
|                | SPATIAL-UNDERSTANDING     | **48.64** | -           | 33.6   | -         |
|                | ATTRIBUTE-RECOGNITION     | **67.08** | -           | 57.2   | -         |
|                | WORLD-KNOWLEDGE           | **53.87** | -           | 50.7   | -         |
|                | OBJECT-LOCALIZATION       | **43.06** | -           | 42.0     | -         |




## General Ability Evaluation: Comparison with LLaVA OneVision-7B and GPT-4

| Dataset     | Split   | MLCD-Embodied-7B | LLaVA OneVision-7B | GPT-4v   | GPT-4o |
| :-- | :-: | :-: | :-: | :-: | :-: |
| A12D        | test    | 79.9             | 81.4               | 78.2     | 94.2   |
| ChartQA     | test    | 83.0             | 80.0               | 78.5     | 85.7   |
| DocVQA      | test    | 91.6             | 87.5               | 88.4     | 92.8   |
| InfoVQA     | val     | 73.9             | 70.7               | -        | -      |
| InfoVQA     | test    | 70.0             | 68.8               | -        | -      |
| MMMU        | val     | 47.3             | 48.8               | 56.8     | 69.1   |
| MMStar      | test    | 58.5             | 61.7               | 57.1     | 63.9   |
| OCRBench    | -       | 749.0            | 697.0              | 656.0    | 805.0  |
| RealWorldQA | test    | 68.9             | 66.3               | 61.4     | 58.6   |
| SeedBench   | image   | 74.9             | 75.4               | 49.9     | 76.2   |
| MMbench     | en-dev  | 81.1             | 83.2               | 81.3     | 83.4   |
| MMbench     | en-test | 80.1             | 80.8               | 75.0     | -      |
| MME         | test    | 578/1603         | 418/1580           | 517/1409 | -      |
