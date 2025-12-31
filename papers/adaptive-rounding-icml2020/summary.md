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
