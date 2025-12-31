# SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot (ICML 2023)

**Idea**: one-shot pruning for billion-scale GPT models using blockwise Hessian-aware reconstruction.

**Mechanism**
- For each block, solve a local quadratic approximation of the loss to choose sparse weights, avoiding
full retraining.
- Works efficiently on very large models (OPT-175B, BLOOM-176B) in a few hours.

**Results**
- Achieves 50–60% unstructured sparsity with negligible perplexity increase.

**Use here**
- Strong pruning baseline; informs sparse + seed hybrid experiments.
