# Research Statement (Transcribed)

Neural network compression tasks:

1. **Post-hoc compression performance without finetuning**
   - Fixed compression rate: 20%
   - Block size = 10 (1 random seed, 1 coefficient)
   - Block size = 20 (1 random seed, 3 coefficients)

2. **Finetune one compressed model from (1)**
   - Finetune without projected gradient descent (full train data)
   - Finetune with projected gradient descent (full train data)
   - Vary projection frequency
   - During projection, try with and without re-optimizing the random seeds

3. **Knowledge distillation with few samples**
   - KD loss; sweep percentages of data

Other notes: NOLA (compress LoRA via random bases), SeedLM (seed-based LLM compression), MCNC (manifold reparameterization), SVD/SVD-LLM, quantization (Quantization Networks, SmoothQuant, Adaptive Rounding), pruning (PLATON, Wanda, SparseGPT).
