# SVD-LLM: Truncation-Aware Singular Value Decomposition for LLM Compression (ICLR 2025)

**Problem**: naive SVD truncation drops small singular values and ignores post-truncation adaptation,
leading to accuracy loss.

**Contributions**
- Truncation-aware objective that anticipates the effect of cutting the tail singular values.
- Post-truncation update of compressed weights (small optimization step) to reduce reconstruction error.
- Demonstrated on LLMs with improved perplexity/zero-shot scores over vanilla SVD at the same rank.

**Use here**
- Strong low-rank baseline to compare with weighted SVD and SeedLM; informs projection frequency for
post-hoc finetuning.
