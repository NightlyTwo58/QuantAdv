"""
QuantAdv/src_old: quantization-aware training + adversarial-robustness evaluation
pipeline for CIFAR-10 models.

Covers torchao-based PTQ/QAT int8 quantization (`quantadv.model`), FGSM/PGD/
AutoAttack/BPDA attacks (`quantadv.attacks`), gradient-masking diagnostics
(`quantadv.diagnostics`), the per-model evaluation suite (`quantadv.suite`),
the experiment runner (`quantadv.experiment`), and result aggregation /
plotting (`quantadv.combine`).
"""
