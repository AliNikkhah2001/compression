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
