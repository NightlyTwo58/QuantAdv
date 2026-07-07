# QuantAdv
Quantized models introduce discrete rounding operations into the computational graph, which may produce either genuine robustnes against *inference-time evasion attacks* (coarser weight representation changing the decision boundary geometry) or gradient masking (rounding causing zero gradients that blind attacks). We systematically evaluate several models (ResNet20, ResNet56 MobileNetV2, VGG16_BN, ShuffleNetV2, and RepVGG_A0) across four quantization variants (FP32, int8 PTQ, int4 PTQ, int8 QAT) on the CIFAR-10 image dataset using a layered attack suite (FGSM, PGD, AutoAttack, transfer attacks from FP32, and BPDA-corrected PGD) to test these two explanations.  

## Setup

To install (most) dependencies  
`pip install -r requirements.txt`

To download datasets (should be placed at root)  
*CIFAR-10*
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

*CIFAR-100*
```
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
curl -O https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

or set Download=true

To run  
`python src/run_experiment.py`  
`python src/combine_results`  
`python archive/QuantAdv.py`  

Parallelized (obsolete)  
`python archive/launcher.py`

> **Notice:** You may need to adjust pathing or move the scripts to root for obsolete files.

Results are in ./data

![Attack Accuracy](demo/accuracy_plot.png)
![Elipson](demo/accuracyperturb.png)

<!-- <p align="center">
  <img src="data/demo/accuracy_plot.png" alt="Attack Accuracy" width="400"/>
  <img src="data/demo/accuracyperturb.png" alt="Elipson" width="400"/>
</p> -->