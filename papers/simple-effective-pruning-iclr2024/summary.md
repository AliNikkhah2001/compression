# A Simple and Effective Pruning Approach for Large Language Models (ICLR 2024)

**Method (Wanda)**: score weights by |w|·|a| (magnitude times activation) for fast one-shot pruning without Hessians or retraining.

**Key points**
- Minimal calibration data; scales to billion-parameter LLMs.
- Outperforms simple magnitude pruning; close to heavier methods at similar sparsity.

**Use here**
- Pruning baseline vs SparseGPT and PLATON; can be combined with seed-based encoding post-prune.
