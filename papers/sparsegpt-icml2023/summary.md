# SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot (ICML 2023)

**Problem**: pruning billion-scale GPT models without retraining.

**Idea**: one-shot pruning using blockwise Hessian-aware reconstruction to choose which weights to drop.

**Method**
- For each block, solve a local quadratic approximation to decide sparse mask.
- No full retraining; runs efficiently even on 100B+ models.

**Findings (paper)**
- 50–60% unstructured sparsity with negligible perplexity increase on OPT/BLOOM.

**Use here**
- Strong pruning baseline; informs sparse+seed hybrid experiments.
