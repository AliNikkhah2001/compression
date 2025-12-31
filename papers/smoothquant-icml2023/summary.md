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
