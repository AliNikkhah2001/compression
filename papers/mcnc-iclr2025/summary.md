# MCNC: Manifold-Constrained Reparameterization for Neural Compression (ICLR 2025)

**Problem**: maintain structure while compressing massive models (e.g., Llama 3.1 405B) without losing accuracy.

**Idea**: encode each weight block as a latent point on a manifold (e.g., Stiefel/Grassmann), decode to weights, and constrain optimization to the manifold.

**Method**
- Manifold-aware reparameterization: latent -> decoder -> weight block.
- Projection operators enforce constraints during updates (PGD-friendly).
- Supports both storage compression and parameter-efficient finetuning.

**Findings (paper)**
- Reduces storage while preserving accuracy on large models.
- More stable finetuning compared to unconstrained low-rank baselines.

**Use here**
- Baseline projection operator for PGD-on-manifold; comparator to SeedLM manifold.
