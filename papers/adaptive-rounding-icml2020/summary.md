# Up or Down? Adaptive Rounding for Post-Training Quantization (ICML 2020)

**Idea**: learn per-weight rounding offsets (AdaRound) to minimize task loss using a small unlabeled
calibration set. Produces better PTQ than nearest-neighbor rounding.

**Mechanism**
- Optimize a continuous relaxation of rounding with an $L_2$ regularizer toward integer points.
- After optimization, snap to discrete values for deployment.

**Results**
- Improves PTQ accuracy across CNNs and transformers without full finetuning.

**Use here**
- PTQ baseline and potential component in seed-based reconstruction (optimize rounding of coefficients).
