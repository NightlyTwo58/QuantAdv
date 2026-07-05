# QuantAdv
Quantized models introduce discrete rounding operations into the computational graph, which may produce two distinct effects: genuine robustness (coarser weight representation changing the decision boundary geometry) or gradient masking (rounding causing near-zero input gradients that blind iterative attacks like PGD, producing illusory robustness without real protection). We systematically evaluate several models (ResNet20, ResNet56 MobileNetV2, VGG16_BN, ShuffleNetV2, and RepVGG_A0) across four quantization variants (FP32, int8 PTQ, int4 PTQ, int8 QAT) on the CIFAR-10 image dataset using a layered attack suite (FGSM, PGD, AutoAttack, transfer attacks from FP32, and BPDA-corrected PGD) to distinguish these two explanations.  
We additionally report gradient diagnostics (fraction of near-zero gradient components under hard vs. STE rounding) and accuracy-vs-epsilon curves across both architectures to characterize whether apparent robustness gains are mechanistically attributable to masking or represent genuine changes in model vulnerability.


To install (most) dependencies  
`pip install -r requirements.txt`

To run  
`python QuantAdv.py`
`python combine_results`

Parallelized (obselete)  
`python launcher.py`

Results are in ./data