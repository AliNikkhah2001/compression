# SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators

**Core idea**: replace stored weights with PRNG seeds and a few coefficients per block. For a block
$W_i \in \mathbb{R}^B$, reconstruct via
$$W_i \approx \sum_{k=1}^{K} \alpha_{i,k} \cdot G(s_{i,k}),$$
where $G$ is an LFSR-based generator seeded by integer $s_{i,k}$ and $K \ll B$.

**Method highlights**
- Data-free, post-training compression: find seeds and coefficients without calibration data.
- Trades compute for memory: generate random bases on-device, reducing memory traffic for memory-bound LLM inference.
- Two variants: fixed seeds + learned coefficients, or joint optimization with seed search.
- Projection step keeps weights on the seed manifold.

**Results (paper)**
- Compresses Llama 2/3 models (incl. 70B) to 3–4 bit equivalents with minimal zero-shot loss; outperforms PTQ baselines at the same bitwidth.
- FPGA tests show ~4× speedup at scale by reducing memory accesses.

**Relevance to this repo**
- Matches our planned block-size sweeps, PGD-on-manifold finetuning, and KD under low-data regimes.
