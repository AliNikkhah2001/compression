# Neural Compression Lab

Research workspace for neural network compression across reparameterization, low-rank/SVD, quantization, pruning, and seed-based weight generation.
_Last updated: 2025-12-31 17:27 UTC_

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

**Problem**: nearest-neighbor rounding is suboptimal for task loss.

**Idea**: learn per-weight rounding offsets (AdaRound) using a small unlabeled calibration set to minimize task loss, then snap to integers.

**Method**
- Continuous relaxation of rounding with regularization toward integers.
- Optimize offsets on calibration data; finalize with hard rounding.

**Findings (paper)**
- Improves PTQ accuracy across CNNs/transformers without full finetuning.

**Use here**
- PTQ baseline; offsets could be applied to SeedLM coefficients.
</details>

- [ ] **MCNC: Manifold-Constrained Reparameterization for Neural Compression (ICLR 2025)** (`papers/mcnc-iclr2025`) — tags: manifold, reparameterization, compression, llm
<details>
<summary>Show details</summary>

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

**Problem**: importance scores from mini-batches are noisy; risk pruning crucial weights.

**Idea**: frame pruning as a bandit; use Upper Confidence Bounds (UCB) to balance estimated importance and uncertainty when deciding what to keep.

**Method**
- Compute weight importance + uncertainty from batch statistics.
- UCB-based selection for pruning steps; supports iterative pruning.

**Findings (paper)**
- Outperforms magnitude pruning at equal sparsity on transformers; benefits from light finetuning.

**Use here**
- Pruning baseline vs SparseGPT and Wanda; potential precursor to seed-encoding sparse weights.
</details>

- [ ] **Quantization Networks (CVPR 2019)** (`papers/quantization-networks-cvpr2019`) — tags: quantization, vision, mixed-precision
<details>
<summary>Show details</summary>

# Quantization Networks (CVPR 2019)

**Problem**: fixed quantizers limit accuracy at low bitwidths.

**Idea**: learn quantization functions end-to-end via a differentiable quantizer network; support mixed precision by learning bit-width/scale per layer.

**Method**
- Soft-to-hard annealing for rounding during training.
- Quantizer parameters trained jointly with model weights.
- Mixed-precision assignment emerges from learning.

**Findings (paper)**
- Competitive accuracy on ResNet/MobileNet at low bits; better than hand-designed PTQ in vision tasks.

**Use here**
- Vision baseline; informs learned quantizers for ViT/DeiT vs SmoothQuant PTQ.
</details>

- [ ] **SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators** (`papers/seedlm`) — tags: seedlm, reparameterization, post-training, llm
<details>
<summary>Show details</summary>

# SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators

**Problem**: LLMs are memory-bound; storing dense weights dominates inference latency/cost.

**Idea**: replace explicit weights with PRNG seeds + a few coefficients per block. For block
$W_i \in \mathbb{R}^B$:
$$W_i \approx \sum_{k=1}^{K} \alpha_{i,k} \cdot G(s_{i,k}),$$
where $G$ is an LFSR-based generator seeded by integer $s_{i,k}$ and $K \ll B$.

**Method**
- Data-free post-training compression; search seeds/coefficients per block.
- Two modes: fixed seeds with learned coefficients, or joint seed/coeff optimization.
- Projection keeps weights on the seed manifold; trades compute for reduced memory traffic.

**Findings (paper)**
- Llama 2/3 (up to 70B) compressed to 3–4 bit equivalents with minimal zero-shot loss.
- Outperforms PTQ baselines at same bitwidth; FPGA tests show ~4× speedup from reduced memory access.

**Limitations/next**
- Seed search cost; discrete optimization.
- Explore hybrid with KD, PGD, quantization of coefficients.
</details>

- [ ] **A Simple and Effective Pruning Approach for Large Language Models (ICLR 2024)** (`papers/simple-effective-pruning-iclr2024`) — tags: pruning, one-shot, llm
<details>
<summary>Show details</summary>

# A Simple and Effective Pruning Approach for Large Language Models (ICLR 2024)

**Method (Wanda)**: score weights by |w|·|a| (magnitude times activation) for fast one-shot pruning without Hessians or retraining.

**Key points**
- Minimal calibration data; scales to billion-parameter LLMs.
- Outperforms simple magnitude pruning; close to heavier methods at similar sparsity.

**Use here**
- Pruning baseline vs SparseGPT and PLATON; can be combined with seed-based encoding post-prune.
</details>

- [ ] **SmoothQuant (ICML 2023)** (`papers/smoothquant-icml2023`) — tags: quantization, ptq, llm
<details>
<summary>Show details</summary>

# SmoothQuant (ICML 2023)

**Problem**: activation outliers make LLM activations hard to quantize; naive PTQ hurts accuracy.

**Idea**: offline smooth activations by shifting per-channel scale into weights so activations become easier to quantize (enabling W8A8 PTQ).

**Method**
- Choose scaling factor per channel; transform $(W, A)$ to $(W / s, s \cdot A)$ before quantization.
- Calibrate scales on a small dataset; quantize weights/activations afterward.

**Findings (paper)**
- Training-free W8A8 on OPT/BLOOM with minimal perplexity increase.
- Hardware-friendly (no dynamic scaling at inference).

**Use here**
- PTQ baseline; activation smoothing potentially helpful for seed-reconstructed weights too.
</details>

- [ ] **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot (ICML 2023)** (`papers/sparsegpt-icml2023`) — tags: pruning, one-shot, hessian, llm
<details>
<summary>Show details</summary>

# SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot (ICML 2023)

**Problem**: pruning billion-scale GPT models without retraining.

**Idea**: one-shot pruning using blockwise Hessian-aware reconstruction to choose which weights to drop.

**Method**
- For each block, solve a local quadratic approximation to decide sparse mask.
- No full retraining; runs efficiently even on 100B+ models.

**Findings (paper)**
- 50–60% unstructured sparsity with negligible perplexity increase on OPT/BLOOM.

**Use here**
- Strong pruning baseline; informs sparse+seed hybrid experiments.
</details>

- [ ] **SVD-LLM: Truncation-Aware Singular Value Decomposition for LLM Compression (ICLR 2025)** (`papers/svd-llm-iclr2025`) — tags: low-rank, svd, llm, post-training
<details>
<summary>Show details</summary>

# SVD-LLM: Truncation-Aware Singular Value Decomposition for LLM Compression (ICLR 2025)

**Problem**: naive SVD truncation discards tail singular values and skips post-truncation adaptation, causing accuracy loss.

**Idea**: optimize with awareness of the truncation boundary and adapt compressed weights afterward.

**Method**
- Truncation-aware objective anticipating dropped singular values.
- Light post-truncation optimization to refine compressed weights.

**Findings (paper)**
- Improves perplexity/zero-shot over vanilla SVD at the same rank on LLMs.

**Use here**
- Strong low-rank baseline vs weighted SVD and SeedLM; informs projection frequency for finetuning.
</details>

- [ ] **Language Model Compression with Weighted Low-Rank Factorization (ICLR 2022)** (`papers/weighted-low-rank-iclr2022`) — tags: low-rank, svd, compression, llm
<details>
<summary>Show details</summary>

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
