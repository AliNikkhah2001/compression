# SVD-LLM: Truncation-Aware Singular Value Decomposition for LLM Compression (ICLR 2025)

**Problem**: Standard truncated SVD drops small singular values without accounting for truncation error; no post-truncation adaptation.

**Setup**
- Let $W = U \\Sigma V^\\top$, singular values $\\sigma_1 \\ge \\dots \\ge \\sigma_p$.
- Truncate to rank $r$: $\\hat W = U_r \\Sigma_r V_r^\\top$ (keep top-$r$).
- Truncation error: $\\|W-\\hat W\\|_F^2 = \\sum_{i>r} \\sigma_i^2$.

**Truncation-aware objective**
- Optimize $U_r,\\Sigma_r,V_r$ anticipating the discarded tail:
  $$\\min_{U_r,\\Sigma_r,V_r} \\; \\mathcal{L}_{task}(U_r \\Sigma_r V_r^\\top) + \\lambda \\sum_{i>r} \\sigma_i^2,$$
  where the second term penalizes tail mass and aligns $r$ with task loss.

**Post-truncation adaptation**
- After initial truncation, run a few gradient steps on $U_r,\\Sigma_r,V_r$ w.r.t. task loss (keeping rank fixed) to compensate for tail removal.
- Equivalent to low-rank finetuning constrained to rank $r$.

**Findings (paper)**
- Lower perplexity/zero-shot loss than vanilla SVD at the same $r$ on LLMs.
- Gains especially when $r$ is small (aggressive compression).

**Use here**
- Strong low-rank baseline vs weighted SVD and SeedLM; informs how often to adapt after projection/truncation.
