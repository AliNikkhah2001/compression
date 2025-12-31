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
