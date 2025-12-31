# MCNC: Manifold-Constrained Reparameterization for Neural Compression (ICLR 2025)

**Idea**: map each weight block to a low-dimensional latent that lives on a constrained manifold, then
reconstruct the weight via a decoder while enforcing manifold geometry (e.g., orthogonality or norm
constraints). This unifies compression and finetuning: optimization happens in latent space while
weights stay on the manifold.

**Key components**
- Manifold parameterization (e.g., Stiefel/Grassmann) to preserve structure during optimization.
- Projection operators to keep updates on-manifold (compatible with PGD-style finetuning).
- Works with large models (e.g., Llama 3.1 405B cited) aiming at aggressive storage reduction.

**Why it matters here**
- Directly aligns with our PGD-on-manifold experiments and seed-based reparameterization; can be used
as a comparator or as a projection operator baseline.
