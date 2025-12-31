# SmoothQuant (ICML 2023)

**Idea**: make activations easier to quantize by smoothing outliers. Moves part of activation scale
into weights offline so that post-training quantization can use W8A8 without accuracy loss.

**Mechanism**
- For each channel, choose scaling factor $s$ and transform $(W, A)$ to $(W / s, s \cdot A)$ prior to
quantization.
- Calibrate scales on a small dataset; quantize weights/activations afterward.

**Results**
- Enables training-free W8A8 on LLMs (OPT/BLOOM etc.) with minimal perplexity increase.
- Hardware-friendly (no dynamic scaling during inference).

**Use here**
- PTQ baseline for our models; informs activation smoothing even for seed-based reconstructions.
