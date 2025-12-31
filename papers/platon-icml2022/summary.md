# PLATON: Pruning Large Transformer Models with UCB of Weight Importance (ICML 2022)

**Idea**: treat weight importance estimation as a multi-armed bandit problem; use Upper Confidence
Bounds (UCB) to balance exploitation (high estimated importance) and exploration (uncertainty) when
selecting weights to keep.

**Highlights**
- Importance estimates from mini-batch statistics with uncertainty-aware pruning decisions.
- Works on transformers, outperforming magnitude pruning at same sparsity.
- Compatible with iterative pruning and light finetuning.

**Use here**
- Pruning baseline for comparison with one-shot sparse methods (SparseGPT) and seed-based sparsity.
