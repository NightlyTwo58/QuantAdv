- **FGSM:** One step in the sign of the input-loss gradient. A fast baseline that searches less than multi-step attacks.
- **PGD:** Repeated gradient steps that raise the loss, projecting each iterate back into the $\epsilon$-ball and valid image range. The runner uses random starts and multiple seeds/restarts. Stronger than FGSM, but can still fail against non-differentiable or poorly behaved gradients.
- **AutoAttack:** A standardized ensemble, not a single attack ([paper](https://proceedings.mlr.press/v119/croce20b.html), [official implementation](https://github.com/fra31/auto-attack)). The configured standard $L_\infty$ version tries these in sequence:
  - **APGD-CE:** Auto-PGD with cross-entropy loss. Adaptive step-size control removes the need to hand-tune the step size.
  - **APGD-DLR:** Auto-PGD with difference-of-logits-ratio loss, which stays useful when cross-entropy gives weak guidance.
  - **FAB:** Fast Adaptive Boundary attack. Seeks a small perturbation toward the decision boundary, with targeted variants in the standard suite.
  - **Square Attack:** Score-based and gradient-free. Proposes localized square-shaped pixel changes, so it works when gradients are unavailable or misleading.

  Adversarial examples found by one component are kept, and the remaining examples go to later components. This reduces reliance on any single loss or gradient method.
- **CW:** Carlini-Wagner-style optimization that balances misclassification against perturbation size. Configured via confidence constant `CW_C`, margin `CW_KAPPA`, steps, and learning rate. Check its wrapper and constraints before comparing results directly to an $L_\infty$ PGD result.
- **DeepFool:** Iteratively estimates the nearest local decision boundary and steps across it. Seeks a minimal perturbation rather than maximizing cross-entropy under a fixed budget like PGD.
- **JSMA:** Jacobian-based saliency attack. Ranks input features by how much changing them pushes the prediction toward a target class, then modifies the top ones. Capped at 20 images by default, so its sample coverage differs from the main attacks.
- **Transfer attack:** Generates adversarial examples on a source model and evaluates those exact images on a target model. Tests whether models share vulnerable directions.
- **MIM transfer:** Momentum Iterative Method on a source model, then transfer. Momentum makes the perturbation direction less sensitive to individual gradient steps and often improves transfer.
- **UAP / UAP transfer:** Builds one universal perturbation meant to fool many examples, instead of one perturbation per image. Disabled by default.
- **Surrogate transfer:** Trains a substitute CNN on labels queried from the target, attacks the substitute, then transfers the examples. Models a limited-query attacker.
- **BPDA-PGD:** PGD that approximates the gradient through a non-differentiable operation. Especially relevant to hard rounding: the forward pass keeps quantization while the backward pass supplies a usable derivative.
- **EOT-PGD:** Averages gradients over repeated forward passes to attack randomized systems. Only useful when randomness is present. Default EOT sample count is 4.
- **Adaptive guardrail / detect-guard attacks:** Attack through the project's guardrail or detector defenses using an objective that accounts for their behavior. Defense-specific, and the code paths are optional.
- **NES:** Estimates a gradient from model outputs on sampled nearby inputs, then iteratively perturbs with it. Needs no backpropagation but uses many queries.
- **Boundary attack:** Query-based search that starts from an adversarial point and shrinks the perturbation while staying adversarial. Not the same as the project's "staircase" diagnostic, which probes behavior around quantization steps.
- **Random noise:** Random perturbations within the configured budget. A control, not an optimized adversary. If it fools the model, the model is vulnerable even to non-targeted noise.

**Attack diagnostics and sweeps** These help interpret attack failures and apparent robustness:
- **PGD step-count ablation:** Checks whether more iterations lower accuracy.
- **PGD trajectory:** Records gradient and movement behavior across steps.
- **Gradient diagnostics:** Compares hard-round gradients against STE gradients.
- **Layerwise profiling:** Measures how gradients propagate through the network.
- **Confidence-margin diagnostics:** Inspects distances between class scores.
- **Epsilon sweep:** Repeats attacks at different budgets $\varepsilon$.