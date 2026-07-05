# QuantAdv
Quantized models introduce discrete rounding operations into the computational graph, which may produce two distinct effects: genuine robustness (coarser weight representation changing the decision boundary geometry) or gradient masking (rounding causing near-zero input gradients that blind iterative attacks like PGD, producing illusory robustness without real protection). We systematically evaluate several models (ResNet20, ResNet56 MobileNetV2, VGG16_BN, ShuffleNetV2, and RepVGG_A0) across four quantization variants (FP32, int8 PTQ, int4 PTQ, int8 QAT) on the CIFAR-10 image dataset using a layered attack suite (FGSM, PGD, AutoAttack, transfer attacks from FP32, and BPDA-corrected PGD) to distinguish these two explanations.  
We additionally report gradient diagnostics (fraction of near-zero gradient components under hard vs. STE rounding) and accuracy-vs-epsilon curves across both architectures to characterize whether apparent robustness gains are mechanistically attributable to masking or represent genuine changes in model vulnerability.

## Setup

To install (most) dependencies  
`pip install -r requirements.txt`

To run  
`python src/run_experiment.py`
`python src/combine_results`

Parallelized (obselete)  
`python archive/launcher.py`

> **Notice:** You may need to adjust pathing or move the scripts to root for the archived files.

Results are in ./data

![Attack Accuracy](demo/accuracy_plot.png)
![Elipson](demo/accuracyperturb.png)

<!-- <p align="center">
  <img src="data/demo/accuracy_plot.png" alt="Attack Accuracy" width="400"/>
  <img src="data/demo/accuracyperturb.png" alt="Elipson" width="400"/>
</p> -->