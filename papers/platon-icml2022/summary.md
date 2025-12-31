# PLATON: Pruning Large Transformer Models with UCB of Weight Importance (ICML 2022)

**Problem**: importance scores from mini-batches are noisy; risk pruning crucial weights.

**Idea**: frame pruning as a bandit; use Upper Confidence Bounds (UCB) to balance estimated importance and uncertainty when deciding what to keep.

**Method**
- Compute weight importance + uncertainty from batch statistics.
- UCB-based selection for pruning steps; supports iterative pruning.

**Findings (paper)**
- Outperforms magnitude pruning at equal sparsity on transformers; benefits from light finetuning.

**Use here**
- Pruning baseline vs SparseGPT and Wanda; potential precursor to seed-encoding sparse weights.
