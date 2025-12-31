# SeedLM Block Search & Reparameterization
[x] Implements block-wise seed-based compression sweeps.

Plan: replicate the PDF tasks—post-hoc compression at fixed 20% rate with varying block sizes (10/20),
seed counts, and coefficients; then finetune with/without projected gradient descent and knowledge
distillation at low-data regimes.
