# Language Model Compression with Weighted Low-Rank Factorization (ICLR 2022)

**Idea**: improve SVD-based compression by weighting reconstruction toward task-critical directions
instead of uniform Frobenius error. The method adds importance weights (from sensitivity metrics) into
the low-rank factorization objective so that singular values aligned with important parameters are
kept.

**Highlights**
- Task-aware weighting bridges the gap between naive SVD and accuracy retention.
- Applicable to transformer matrices (attention/FFN) with minimal code changes.
- Evaluated on language modeling perplexity; yields better accuracy at the same rank than vanilla SVD.

**Use here**
- Baseline for low-rank vs seed-based compression; can serve as KD teacher or initialization.
