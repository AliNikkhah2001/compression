# Experiment Plan: SeedLM & Baselines

## 1) Post-hoc SeedLM compression (no finetune)
- Compression rate: 20% effective bitrate.
- Block sizes: 10 and 20.
- Seeds/coefficients per block: (1 seed,1 coeff) and (1 seed,3 coeff).
- Metrics: reconstruction error, perplexity (LLM), top-1 (vision), latency/bandwidth.

## 2) SeedLM finetuning on manifold
- Start from best post-hoc config above.
- Finetune variants: (a) unconstrained, (b) PGD on seed manifold.
- Projection frequency sweep: every 1/5/10 steps.
- Seed policy: fixed vs re-optimized during projection.
- Metrics: perplexity/accuracy, stability across runs.

## 3) Knowledge distillation (low-data)
- Teacher: full model; Student: compressed SeedLM.
- Data regimes: 1%, 5%, 10% of training data.
- Loss: KD (temperature, soft labels) + optional CE.
- Compare to full-data finetuning.

## 4) LoRA adapter compression (NOLA concept)
- Represent LoRA updates as PRNG basis + coeffs.
- Compare to standard LoRA and base SeedLM compression.

## 5) Baselines and hybrids
- Low-rank: weighted SVD, SVD-LLM (same ranks as SeedLM bitrate).
- Quantization: SmoothQuant (W8A8), AdaRound (lower-bit), Quantization Networks (learned).
- Pruning: SparseGPT (one-shot Hessian), Wanda (|w|·|a|), PLATON (UCB).
- Hybrids: prune then seed-encode; quantize SeedLM coefficients; KD vs PGD interactions.

## 6) Datasets and metrics
- LLM: WikiText-2/3, PTB (perplexity); small calibration sets for PTQ.
- Vision: ImageNet/CIFAR/COCO; calibration splits for PTQ.
- Report: compression ratio, latency/bandwidth, accuracy/perplexity, ablation tables.
