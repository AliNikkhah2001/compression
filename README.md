# Neural Compression Lab

Research workspace for neural network compression across reparameterization, low-rank/SVD, quantization, pruning, and seed-based weight generation.
_Last updated: 2025-12-31 17:35 UTC_

Project homepage: https://github.com/AliNikkhah2001/compression

## What this template includes
- Opinionated folder layout for code, datasets, papers, models, and experiments.
- Metadata-first README generator (`scripts/build_readme.py`).
- Optional GitHub Actions to auto-refresh README and Pages.
- Space for notes, TODOs, and evaluation summaries without cluttering the root README.

## Structure
- `code/`: experiments, pipelines, and libraries
- `datasets/`: dataset manifests, download scripts, and notes
- `papers/`: literature summaries and (optional) PDFs
- `training/`: reproducible training pipelines and configs
- `models/`: lightweight model cards and release links
- `evaluation/`: benchmarks, metrics, and analysis notebooks
- `ideas/`: backlog items, design docs, and hypotheses
- `scripts/`: automation such as the README generator

## Code modules
- [ ] **SeedLM Block Search & Reparameterization** (`code/seedlm-block-search`) — tags: seedlm, reparameterization, compression
<details>
<summary>Show details</summary>

# SeedLM Block Search & Reparameterization
[x] Implements block-wise seed-based compression sweeps and finetuning.

**Goal**: operationalize the PDF tasks—pure post-hoc SeedLM compression at 20% rate, block sizes 10 vs
20, varying seeds/coefficients; then finetune with/without PGD and with KD at low-data.

**Planned steps**
- Build block generator (LFSR/PRNG) + coefficient solver; evaluate reconstruction error vs accuracy.
- Post-hoc evaluation (no finetune) on LLM perplexity and vision accuracy.
- PGD-on-manifold finetuning, sweeping projection frequency; compare fixed vs re-optimized seeds.
- KD distillation at 1/5/10% data; measure recovery vs full-data finetune.
</details>

## Datasets
- [ ] **LLM Compression Benchmarks** (`datasets/llm-compression-benchmarks`) — tags: llm, language, quantization, pruning
<details>
<summary>Show details</summary>

# LLM Compression Benchmarks
[x] Datasets used across LLM compression, quantization, pruning, and distillation.

**Included targets**
- **C4 / RedPajama** (pretrain-scale; for reconstruction/ptq calibration at scale; external download).
- **WikiText-2/3, PTB** (perplexity eval; small calibration subsets for PTQ/AdaRound/SmoothQuant).
- **Calibration splits**: a few thousand tokens for activation smoothing and rounding optimization.
- **KD subsets**: 1%, 5%, 10% slices for low-data distillation experiments.

**Notes**
- Large corpora are *not* stored in git; add manifests + download scripts with checksums.
- Track licensing (C4/RedPajama) and usage constraints.
</details>

- [ ] **Vision Compression Benchmarks** (`datasets/vision-compression-benchmarks`) — tags: vision, imagenet, cifar, quantization
<details>
<summary>Show details</summary>

# Vision Compression Benchmarks
[x] Datasets for vision-side compression baselines (quantization, pruning, low-rank, SeedLM-for-ViT).

**Included targets**
- **ImageNet-1k** (classification; full-scale; external download credentials required).
- **CIFAR-10/100** (fast iteration for QAT/PTQ, pruning, low-rank).
- **COCO keypoints/detection** when testing ViT/DeiT backbones and structured sparsity.
- **Calibration subsets** for PTQ (e.g., 512–2048 images).

**Notes**
- No raw data in git; provide scripts/manifests and document expected directory layout.
- Useful for transferring lessons from Quantization Networks (CVPR 2019) to ViT.
</details>

## Papers
- [ ] **Up or Down? Adaptive Rounding for Post-Training Quantization (ICML 2020)** (`papers/adaptive-rounding-icml2020`) — tags: quantization, ptq, rounding
<details>
<summary>Show details</summary>

# Up or Down? Adaptive Rounding for Post-Training Quantization (ICML 2020)

**Problem**: Nearest rounding $\\mathrm{round}(x/\\Delta)$ is not task-optimal.

**AdaRound formulation**
- For weight $w$, learn a rounding offset $\\alpha$ in
  $$\\tilde w = \\Delta \\cdot \\mathrm{round}\\big(\\tfrac{w}{\\Delta} + \\alpha\\big) - \\Delta \\alpha,$$
  where $\\alpha \\in [0,1]$ (per-weight or per-block).
- Relax rounding with a sigmoid during optimization:
  $$\\mathrm{round}_\\tau(z) = z - \\sigma_\\tau(z-\\lfloor z \\rfloor - 0.5) + 0.5,$$
  temperature $\\tau \\downarrow 0$.

**Optimization**
- Minimize calibration loss:
  $$\\min_{\\alpha} \\; \\mathcal{L}_{calib}(Q_\\alpha(W); \\text{data}) + \\lambda \\sum \\mathrm{reg}(\\alpha),$$
  where $Q_\\alpha$ uses learned rounding; regularizer pulls $\\alpha$ toward integers (0 or 1).
- After convergence, snap to hard rounding with learned offsets.

**Findings (paper)**
- Improves PTQ accuracy on CNNs/transformers vs nearest rounding; no full finetuning required.

**Use here**
- PTQ baseline; same offset idea could tune SeedLM coefficients when quantized.
</details>

- [ ] **MCNC: Manifold-Constrained Reparameterization for Neural Compression (ICLR 2025)** (`papers/mcnc-iclr2025`) — tags: manifold, reparameterization, compression, llm
<details>
<summary>Show details</summary>

# MCNC: Manifold-Constrained Reparameterization for Neural Compression (ICLR 2025)

**Problem**: Compress massive transformers while keeping optimization stable by enforcing geometric structure on compressed parameters.

**Parameterization**
- Each weight block $W \\in \\mathbb{R}^{m\\times n}$ is mapped from a latent $Z$ that lies on a manifold (e.g., Stiefel manifold of orthonormal columns, Grassmann).
- Decoder $f_\\theta$ reconstructs weights: $\\hat W = f_\\theta(Z)$ with constraints $Z \\in \\mathcal{M}$.

**Objective**
- Reconstruction/compression loss:
  $$\\min_{Z \\in \\mathcal{M},\\,\\theta} \\; \\|W - f_\\theta(Z)\\|_F^2 + \\lambda\\,\\mathcal{R}(Z,\\theta),$$
  where $\\mathcal{R}$ may encourage low-rank or sparsity in the decoded weights.
- During finetuning, task loss $\\mathcal{L}_{task}(f_\\theta(Z))$ replaces reconstruction.

**Manifold optimization**
- Use Riemannian gradient steps on $Z$ with retraction $\\mathrm{Retr}_Z$:
  $$Z \\leftarrow \\mathrm{Retr}_Z(-\\eta \\, \\mathrm{grad}_Z \\mathcal{L}).$$
- Projection keeps $Z$ on $\\mathcal{M}$ (PGD-compatible), improving stability over unconstrained updates.

**Key effects**
- Implicit regularization: orthogonality/normalization in $Z$ controls Lipschitzness of $\\hat W$.
- Compression: storage is latent $Z$ (low-dim) + decoder parameters (shared).

**Findings (paper)**
- Storage reduction on large Llama variants with minimal accuracy loss.
- More stable finetuning than unconstrained low-rank baselines.

**Use here**
- PGD projection operator baseline for SeedLM manifold comparisons; could reuse retraction ops for seed-coefficient manifolds.
</details>

- [ ] **NOLA: Compressing LoRA Using Linear Combination of Random Bases** (`papers/nola-lora-compression`) — tags: lora, seedlm, adapter, compression
<details>
<summary>Show details</summary>

# NOLA: Compressing LoRA Using Linear Combination of Random Bases

**Concept (from research statement)**: compress LoRA adapters by expressing each low-rank update as a
linear combination of PRNG-generated basis matrices plus small coefficients, mirroring SeedLM for the
adapter weights.

**Potential benefits**
- Further storage reduction over standard LoRA.
- PRNG generation can trade compute for memory on-device.

**Next steps**
- Locate paper/preprint or internal notes.
- Prototype random-basis LoRA adapter compression and compare to vanilla LoRA and SeedLM on the base weights.
</details>

- [ ] **PLATON: Pruning Large Transformer Models with UCB of Weight Importance (ICML 2022)** (`papers/platon-icml2022`) — tags: pruning, transformer, importance
<details>
<summary>Show details</summary>

# PLATON: Pruning Large Transformer Models with UCB of Weight Importance (ICML 2022)

**Problem**: Importance estimates from minibatches are noisy; naive magnitude pruning may drop crucial weights.

**Bandit/UCB framing**
- For each weight (or block) $w_j$, maintain estimate $\\hat \\mu_j$ (importance) and uncertainty $\\hat \\sigma_j$ from minibatch statistics.
- Upper Confidence Bound score:
  $$\\mathrm{UCB}_j = \\hat \\mu_j + \\beta \\hat \\sigma_j,$$
  with exploration parameter $\\beta$.
- Prune weights with lowest UCB, balancing exploitation (high mean) and exploration (uncertainty).

**Importance estimation**
- Often derived from gradients/activations: e.g., $\\hat \\mu_j \\propto \\mathbb{E}[|w_j \\cdot a_j|]$ or Fisher/Hessian diagonals; $\\hat \\sigma_j$ from batch variance.

**Procedure**
1) Collect stats over calibration batches.
2) Compute $\\mathrm{UCB}_j$.
3) Prune lowest-scoring weights to target sparsity (iterative or one-shot).
4) Optional light finetuning.

**Findings (paper)**
- Outperforms magnitude pruning at equal sparsity on transformers; robustness from uncertainty term.

**Use here**
- Pruning baseline vs SparseGPT/Wanda; can precede seed-based encoding on the remaining weights.
</details>

- [ ] **Quantization Networks (CVPR 2019)** (`papers/quantization-networks-cvpr2019`) — tags: quantization, vision, mixed-precision
<details>
<summary>Show details</summary>

# Quantization Networks (CVPR 2019)

**Problem**: Fixed, hand-crafted quantizers and uniform bitwidths can underfit at low precision.

**Learned quantizer parameterization**
- Quantizer $Q_\\phi(x)$ with learnable scale/offset/bitwidth; forward uses
  $$\\tilde x = \\mathrm{clip}\\Big(\\mathrm{round}\\big(\\tfrac{x}{\\Delta}\\big), q_{min}, q_{max}\\Big), \\quad Q_\\phi(x) = \\Delta \\tilde x,$$
  where $\\Delta$ (and sometimes $q_{min},q_{max}$ or bitwidth) are learnable via a small network.

**Soft-to-hard annealing**
- Replace $\\mathrm{round}(\\cdot)$ with a smooth approximation $\\mathrm{round}_\\tau$ during training, temperature $\\tau \\downarrow 0$ so gradients flow early, converge to hard quantization late.
- Straight-through estimators (STE) used for backprop through $\\mathrm{round}$ in later stages.

**Mixed precision**
- Bitwidth selection can be parameterized with Gumbel-Softmax over candidate bits; loss includes complexity regularizer:
  $$\\mathcal{L} = \\mathcal{L}_{task} + \\lambda \\sum_\\ell c(b_\\ell),$$
  where $b_\\ell$ is sampled/relaxed bitwidth for layer $\\ell$ and $c(\\cdot)$ penalizes high bits.

**Findings (paper)**
- On ResNet/MobileNet, learned quantizers match or beat hand-designed PTQ at 2–4 bits.
- Learns non-uniform step sizes where beneficial; mixed precision emerges automatically.

**Use here**
- Vision baseline; ideas transferable to ViT/DeiT and to learned coefficient quantization for SeedLM.
</details>

- [ ] **SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators** (`papers/seedlm`) — tags: seedlm, reparameterization, post-training, llm
<details>
<summary>Show details</summary>

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
</details>

- [ ] **A Simple and Effective Pruning Approach for Large Language Models (ICLR 2024)** (`papers/simple-effective-pruning-iclr2024`) — tags: pruning, one-shot, llm
<details>
<summary>Show details</summary>

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
</details>

- [ ] **SmoothQuant (ICML 2023)** (`papers/smoothquant-icml2023`) — tags: quantization, ptq, llm
<details>
<summary>Show details</summary>

# SmoothQuant (ICML 2023)

**Problem**: Activation outliers in LLMs make activations hard to quantize; naive PTQ at W8A8 degrades accuracy.

**Key idea**: Move per-channel activation scale into weights offline, “smoothing” activations.

**Math**
- For a linear layer $y = W x$, per-channel (or per-row) scales $s$:
  $$W' = W / s, \\quad x' = s \\odot x,$$
  so $y = W'x'$ but activations $x'$ have reduced dynamic range.
- Choose $s$ by minimizing post-quantization error on a calibration set:
  $$\\min_s \\; \\| Q_W(W/s) Q_A(s \\odot x) - W x \\|_2^2,$$
  where $Q_W, Q_A$ are weight/activation quantizers (e.g., 8-bit).

**Procedure**
1) Collect small calibration set (few hundred/thousand tokens).
2) Estimate $s$ per channel via optimization/closed-form heuristics.
3) Quantize $W', x'$ with standard INT8.

**Findings (paper)**
- Enables training-free W8A8 on OPT/BLOOM with minimal perplexity change.
- No runtime overhead (scales folded into weights).

**Use here**
- PTQ baseline; smoothing can also stabilize activation ranges for seed-reconstructed weights before quantization.
</details>

- [ ] **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot (ICML 2023)** (`papers/sparsegpt-icml2023`) — tags: pruning, one-shot, hessian, llm
<details>
<summary>Show details</summary>

# SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot (ICML 2023)

**Problem**: Prune 100B-scale GPT models in one shot, no retraining, minimal perplexity loss.

**Hessian-aware pruning (per block)**
- Approximate loss with quadratic form using Hessian $H$ and gradient $g$:
  $$\\mathcal{L}(W + \\Delta W) \\approx \\mathcal{L}(W) + g^\\top \\Delta W + \\tfrac{1}{2} \\Delta W^\\top H \\Delta W.$$
- Goal: choose sparse mask $M$ to minimize quadratic reconstruction:
  $$\\min_{M} \\; \\| H^{1/2} (W - M \\odot W) \\|_F^2,$$
  where $M \\odot W$ keeps selected weights. Often solved greedily per column/row with local LS updates given $H$.

**Procedure**
1) Estimate blockwise Hessian (diagonal/low-rank) from a small calibration set.
2) For each block: select weights to keep by minimizing quadratic error; drop the rest (one-shot).
3) Assemble sparse model; optional tiny finetune (usually skipped).

**Findings (paper)**
- Achieves ~50–60% unstructured sparsity on OPT/BLOOM with negligible perplexity increase.
- Runs efficiently on 100B+ models (few hours).

**Use here**
- Strong pruning baseline; candidate precursor to seed-encoding the retained weights (sparse+seed hybrid).
</details>

- [ ] **SVD-LLM: Truncation-Aware Singular Value Decomposition for LLM Compression (ICLR 2025)** (`papers/svd-llm-iclr2025`) — tags: low-rank, svd, llm, post-training
<details>
<summary>Show details</summary>

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
</details>

- [ ] **Language Model Compression with Weighted Low-Rank Factorization (ICLR 2022)** (`papers/weighted-low-rank-iclr2022`) — tags: low-rank, svd, compression, llm
<details>
<summary>Show details</summary>

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
</details>

## Models & Training
- Document experiment configs and link to checkpoints or releases.

## Evaluation
- Add benchmark summaries, result tables, or notebooks here.

## Ideas & TODOs
- Research Statement (ideas/research-statement.md)
- Todo (ideas/todo.md)

## Using this template
1) Edit `project.json` with your project name/description.
2) Add code/dataset/paper folders with `summary.md`, `tags.md`, and optional `status.md` checkboxes.
3) Run `python3 scripts/build_readme.py` to regenerate `README.md` and `docs/index.md`.
4) Commit your changes; GitHub Actions will refresh the docs on push.

The generator keeps the README short while surfacing progress across modules, datasets, and readings.
