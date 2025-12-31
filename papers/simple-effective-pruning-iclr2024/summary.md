# A Simple and Effective Pruning Approach for Large Language Models (ICLR 2024)

**Method (Wanda)**: One-shot pruning with importance = elementwise product of weight magnitude and activation magnitude.

**Scoring**
- For weight tensor $W$ with activation $A$ (from a small calibration set), score each element:
  $$s_{ij} = |W_{ij}| \\cdot \\mathbb{E}[|A_j|],$$
  or layerwise variants (normalize by channel norms).
- Sort $s_{ij}$ and prune the lowest scores to target sparsity.

**Why it works**
- Incorporates both parameter size and signal magnitude; cheap (no Hessian).
- Calibration activations capture runtime salience.

**Procedure**
1) Run a few batches to gather $\\mathbb{E}[|A|]$.
2) Compute $s_{ij}$, prune lowest-scoring weights to target sparsity.
3) Optional light finetuning (often skipped/short).

**Findings (paper)**
- Scales to billion-parameter LLMs; outperforms magnitude pruning.
- Competitive with heavier methods at similar sparsity.

**Use here**
- Pruning baseline vs SparseGPT/PLATON; can be combined with SeedLM encoding post-prune.
