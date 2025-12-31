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
