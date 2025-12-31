# SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators

**Problem**: LLMs are memory-bound; storing dense weights dominates inference latency/cost.

**Idea**: replace explicit weights with PRNG seeds + a few coefficients per block. For block
$W_i \in \mathbb{R}^B$:
$$W_i \approx \sum_{k=1}^{K} \alpha_{i,k} \cdot G(s_{i,k}),$$
where $G$ is an LFSR-based generator seeded by integer $s_{i,k}$ and $K \ll B$.

**Method**
- Data-free post-training compression; search seeds/coefficients per block.
- Two modes: fixed seeds with learned coefficients, or joint seed/coeff optimization.
- Projection keeps weights on the seed manifold; trades compute for reduced memory traffic.

**Findings (paper)**
- Llama 2/3 (up to 70B) compressed to 3–4 bit equivalents with minimal zero-shot loss.
- Outperforms PTQ baselines at same bitwidth; FPGA tests show ~4× speedup from reduced memory access.

**Limitations/next**
- Seed search cost; discrete optimization.
- Explore hybrid with KD, PGD, quantization of coefficients.
