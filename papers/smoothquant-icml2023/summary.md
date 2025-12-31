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
