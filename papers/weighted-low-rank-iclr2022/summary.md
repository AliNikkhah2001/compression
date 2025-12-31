# Language Model Compression with Weighted Low-Rank Factorization (ICLR 2022)

**Problem**: Plain SVD minimizes $\\|W-UV^\\top\\|_F^2$, treating all entries equally; task-critical directions may be underfit.

**Weighted objective**
- Given importance weights $M \\in \\mathbb{R}^{m\\times n}$ (from sensitivity/gradient/Hessian proxies), solve
  $$\\min_{U,V,\\,\\mathrm{rank}=r} \\; \\| M \\odot (W - U V^\\top) \\|_F^2,$$
  where $\\odot$ is elementwise product.
- When $M=\\mathbf{1}$, recovers vanilla SVD.

**Solution sketch**
- Reweight rows/cols: define $W' = \\sqrt{M} \\odot W$; approximate via truncated SVD of $W'$.
- Optionally iterate: update $M$ from new sensitivities, refit $U,V$.

**Metrics**
- Rank $r$ controls compression (params $r(m+n)$ vs $mn$).
- Evaluate perplexity/accuracy vs $r$; weighted loss yields lower task loss at fixed $r$.

**Findings (paper)**
- On transformer layers, weighted SVD beats vanilla SVD at same rank (better perplexity).
- Minimal code change relative to standard truncated SVD workflow.

**Use here**
- Low-rank baseline vs SVD-LLM and SeedLM; possible teacher/init for hybrid methods.
