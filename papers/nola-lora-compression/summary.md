# NOLA: Compressing LoRA Using Linear Combination of Random Bases

**Concept (from research statement)**: compress LoRA adapters by expressing each low-rank update as a
linear combination of PRNG-generated basis matrices plus small coefficients, mirroring SeedLM for the
adapter weights.

**Potential benefits**
- Further storage reduction over standard LoRA.
- PRNG generation can trade compute for memory on-device.

**Next steps**
- Locate paper/preprint or internal notes.
- Prototype random-basis LoRA adapter compression and compare to vanilla LoRA and SeedLM on the base weights.
