# Quantization Networks (CVPR 2019)

**Idea**: learn quantization functions end-to-end with a differentiable quantizer parameterized by a
small network. Supports mixed-precision by learning bit-width/scale per layer while training the main
model.

**Highlights**
- Soft-to-hard annealing for rounding during training.
- Jointly optimizes quantization parameters and model weights; no separate PTQ step.
- Evaluated on vision models (ResNet, MobileNet) showing competitive accuracy at low bitwidths.

**Use here**
- Vision-side baseline; informs activation smoothing and learnable quantizers for ViT/DeiT when we
compare to SmoothQuant-style PTQ.
