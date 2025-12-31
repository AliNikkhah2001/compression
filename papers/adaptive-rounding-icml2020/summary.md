# Up or Down? Adaptive Rounding for Post-Training Quantization (ICML 2020)

**Problem**: nearest-neighbor rounding is suboptimal for task loss.

**Idea**: learn per-weight rounding offsets (AdaRound) using a small unlabeled calibration set to minimize task loss, then snap to integers.

**Method**
- Continuous relaxation of rounding with regularization toward integers.
- Optimize offsets on calibration data; finalize with hard rounding.

**Findings (paper)**
- Improves PTQ accuracy across CNNs/transformers without full finetuning.

**Use here**
- PTQ baseline; offsets could be applied to SeedLM coefficients.
