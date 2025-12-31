# SeedLM Block Search & Reparameterization
[x] Implements block-wise seed-based compression sweeps and finetuning.

**Goal**: operationalize the PDF tasks—pure post-hoc SeedLM compression at 20% rate, block sizes 10 vs
20, varying seeds/coefficients; then finetune with/without PGD and with KD at low-data.

**Planned steps**
- Build block generator (LFSR/PRNG) + coefficient solver; evaluate reconstruction error vs accuracy.
- Post-hoc evaluation (no finetune) on LLM perplexity and vision accuracy.
- PGD-on-manifold finetuning, sweeping projection frequency; compare fixed vs re-optimized seeds.
- KD distillation at 1/5/10% data; measure recovery vs full-data finetune.
