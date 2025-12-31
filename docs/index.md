---
layout: default
---

# Neural Compression Lab

Research workspace for neural network compression across reparameterization, low-rank/SVD, quantization, pruning, and seed-based weight generation.
_Last updated: 2025-12-31 17:01 UTC_

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
  - Plan: replicate the PDF tasks—post-hoc compression at fixed 20% rate with varying block sizes (10/20), seed counts, and coefficients; then finetune with/without projected gradient descent and knowledge distillation at low-data regimes.

## Datasets
- [ ] **LLM Compression Benchmarks** (`datasets/llm-compression-benchmarks`) — tags: llm, language, quantization, pruning
  - Includes C4/RedPajama for pretrain-scale evaluation, WikiText-2/3 and PTB for perplexity, and calibration splits for post-training quantization (SmoothQuant, AWQ-style), plus small KD subsets for low-data distillation experiments.
- [ ] **Vision Compression Benchmarks** (`datasets/vision-compression-benchmarks`) — tags: vision, imagenet, cifar, quantization
  - Use ImageNet-1k for full-scale tests, CIFAR-10/100 for rapid iterations, and COCO keypoints/detection when evaluating transformer backbones (e.g., ViT compression). Include calibration subsets for post-training quantization and sparse benchmarks mirroring Quantization Networks (CVPR 2019).

## Papers
- [ ] **Up or Down? Adaptive Rounding for Post-Training Quantization (ICML 2020)** (`papers/adaptive-rounding-icml2020`) — tags: quantization, ptq, rounding
  - **Idea**: learn per-weight rounding offsets (AdaRound) to minimize task loss using a small unlabeled calibration set. Produces better PTQ than nearest-neighbor rounding.
- [ ] **MCNC: Manifold-Constrained Reparameterization for Neural Compression (ICLR 2025)** (`papers/mcnc-iclr2025`) — tags: manifold, reparameterization, compression, llm
  - **Idea**: map each weight block to a low-dimensional latent that lives on a constrained manifold, then reconstruct the weight via a decoder while enforcing manifold geometry (e.g., orthogonality or norm constraints). This unifies compression and finetuning: optimization happens in latent space while weights stay on the manifold.
- [ ] **NOLA: Compressing LoRA Using Linear Combination of Random Bases** (`papers/nola-lora-compression`) — tags: lora, seedlm, adapter, compression
  - Concept note from research statement: further compress LoRA adapters by representing each low-rank update as a linear combination of random basis matrices (seeded generators) plus small coefficients. Aligns with SeedLM philosophy, targeting adapter storage reduction.
- [ ] **PLATON: Pruning Large Transformer Models with UCB of Weight Importance (ICML 2022)** (`papers/platon-icml2022`) — tags: pruning, transformer, importance
  - **Idea**: treat weight importance estimation as a multi-armed bandit problem; use Upper Confidence Bounds (UCB) to balance exploitation (high estimated importance) and exploration (uncertainty) when selecting weights to keep.
- [ ] **Quantization Networks (CVPR 2019)** (`papers/quantization-networks-cvpr2019`) — tags: quantization, vision, mixed-precision
  - **Idea**: learn quantization functions end-to-end with a differentiable quantizer parameterized by a small network. Supports mixed-precision by learning bit-width/scale per layer while training the main model.
- [ ] **SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators** (`papers/seedlm`) — tags: seedlm, reparameterization, post-training, llm
  - **Core idea**: replace stored weights with PRNG seeds and a few coefficients per block. For a block $W_i \in \mathbb{R}^B$, reconstruct via $$W_i \approx \sum_{k=1}^{K} \alpha_{i,k} \cdot G(s_{i,k}),$$ where $G$ is an LFSR-based generator seeded by integer $s_{i,k}$ and $K \ll B$.
- [ ] **A Simple and Effective Pruning Approach for Large Language Models (ICLR 2024)** (`papers/simple-effective-pruning-iclr2024`) — tags: pruning, one-shot, llm
  - **Method (Wanda)**: scores weights by element-wise product of weight magnitude and activation magnitude (importance = |w| · |a|), enabling fast one-shot pruning without expensive Hessian computation or retraining.
- [ ] **SmoothQuant (ICML 2023)** (`papers/smoothquant-icml2023`) — tags: quantization, ptq, llm
  - **Idea**: make activations easier to quantize by smoothing outliers. Moves part of activation scale into weights offline so that post-training quantization can use W8A8 without accuracy loss.
- [ ] **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot (ICML 2023)** (`papers/sparsegpt-icml2023`) — tags: pruning, one-shot, hessian, llm
  - **Idea**: one-shot pruning for billion-scale GPT models using blockwise Hessian-aware reconstruction.
- [ ] **SVD-LLM: Truncation-Aware Singular Value Decomposition for LLM Compression (ICLR 2025)** (`papers/svd-llm-iclr2025`) — tags: low-rank, svd, llm, post-training
  - **Problem**: naive SVD truncation drops small singular values and ignores post-truncation adaptation, leading to accuracy loss.
- [ ] **Language Model Compression with Weighted Low-Rank Factorization (ICLR 2022)** (`papers/weighted-low-rank-iclr2022`) — tags: low-rank, svd, compression, llm
  - **Idea**: improve SVD-based compression by weighting reconstruction toward task-critical directions instead of uniform Frobenius error. The method adds importance weights (from sensitivity metrics) into the low-rank factorization objective so that singular values aligned with important parameters are kept.

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
