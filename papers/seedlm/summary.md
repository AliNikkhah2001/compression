# SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators

**Problem**: Memory bandwidth dominates LLM inference. Storing explicit FP16 weights is costly; want extreme compression with minimal accuracy loss.

**Parameterization**
- Partition a weight matrix into blocks $W_i \in \mathbb{R}^{B \\times C}$ (flatten to $\\mathbb{R}^{BC}$).
- For each block, generate $K$ random basis matrices from seeds $s_{i,k}$ using an LFSR PRNG $G$:
  $$\\mathbf{G}_{i,k} = G(s_{i,k}) \\in \\mathbb{R}^{B \\times C}.$$
- Reconstruct via linear combination:
  $$\\hat W_i = \\sum_{k=1}^{K} \\alpha_{i,k} \\mathbf{G}_{i,k},$$
  where $\\alpha_{i,k}$ are learned coefficients (often low-precision).
- Storage cost per block: $K$ seeds (ints) + $K$ coefficients (low-bit) vs $BC$ floats (huge reduction).

**Objective (data-free PTQ)**
- Minimize reconstruction error per block:
  $$\\min_{\\alpha, s} \\; \\|W_i - \\hat W_i(\\alpha, s)\\|_F^2.$$
- Two regimes:
  1) **Fixed seeds, solve coefficients**: $s$ chosen (e.g., random or small search); $\\alpha$ solved by least squares or small optimization.
  2) **Joint search**: discrete search over $s$, continuous solve for $\\alpha$; often alternating.

**Projected finetuning**
- During downstream finetune, apply PGD to stay on the seed manifold:
  1) Gradient step on $\\alpha$ (and optionally relaxed $s$).
  2) Projection: snap back to nearest seed/coeff representation (e.g., recompute $\\alpha$ by LS given $s$, or reselect $s$ from a candidate pool).
- Optionally re-optimize seeds every $k$ steps to expand expressivity.

**Hardware/throughput rationale**
- Replace memory reads of dense weights with on-chip generation: compute-heavy but bandwidth-light.
- LFSR generation is cheap; coefficients can be quantized (e.g., 3–4 bits).
- Effective “bitrate” comparable to 3–4 bit quantization but with better accuracy retention (per paper).

**Reported results**
- Llama 2/3 up to 70B: 3–4 bit equivalent compression with near-FP16 zero-shot scores.
- Outperforms PTQ baselines (SmoothQuant/AWQ) at same nominal bitwidth, especially at larger models.
- FPGA eval: up to ~4× speedup at scale from memory-traffic reduction.

**Limitations / open items**
- Seed search cost (discrete optimization); block size $B$ trades search vs approximation quality.
- PRNG correlations could hurt very small blocks; may need orthogonalization tricks.
- Coefficient quantization/rounding sensitivity; potential to combine with AdaRound-style optimizers.
