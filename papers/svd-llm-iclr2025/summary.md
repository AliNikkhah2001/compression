# SVD-LLM: Truncation-Aware Singular Value Decomposition for LLM Compression (ICLR 2025)

**Problem**: naive SVD truncation discards tail singular values and skips post-truncation adaptation, causing accuracy loss.

**Idea**: optimize with awareness of the truncation boundary and adapt compressed weights afterward.

**Method**
- Truncation-aware objective anticipating dropped singular values.
- Light post-truncation optimization to refine compressed weights.

**Findings (paper)**
- Improves perplexity/zero-shot over vanilla SVD at the same rank on LLMs.

**Use here**
- Strong low-rank baseline vs weighted SVD and SeedLM; informs projection frequency for finetuning.
