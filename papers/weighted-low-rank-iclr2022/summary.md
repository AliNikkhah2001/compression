# Language Model Compression with Weighted Low-Rank Factorization (ICLR 2022)

**Problem**: vanilla SVD minimizes uniform Frobenius error, ignoring task sensitivity.

**Idea**: introduce importance weights into the low-rank objective so reconstruction favors task-critical directions.

**Method**
- Compute sensitivity/importance per weight (e.g., gradient/Hessian proxies) to weight reconstruction loss.
- Perform weighted SVD/truncated factorization respecting those weights.

**Findings (paper)**
- Better perplexity/accuracy at same rank than vanilla SVD on transformer layers.
- Minimal code change relative to SVD baselines.

**Use here**
- Low-rank baseline vs SVD-LLM and SeedLM; potential KD teacher or init.
