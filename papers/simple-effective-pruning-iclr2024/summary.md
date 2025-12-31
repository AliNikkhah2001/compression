# A Simple and Effective Pruning Approach for Large Language Models (ICLR 2024)

**Method (Wanda)**: scores weights by element-wise product of weight magnitude and activation
magnitude (importance = |w| · |a|), enabling fast one-shot pruning without expensive Hessian
computation or retraining.

**Highlights**
- Works at billion-parameter scale; minimal calibration data.
- Outperforms magnitude pruning and matches heavier methods while being simple.

**Use here**
- Baseline sparse method to compare with SparseGPT and PLATON; potential initialization before seed
compression of sparse weights.
