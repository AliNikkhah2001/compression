# Training & Experiments

Keep reproducible training pipelines, experiment configs, and notebooks here.

Suggested targets:
- **LLMs**: Llama 2/3, OPT, BLOOM/RedPajama; measure perplexity on WikiText/PTB and zero-shot tasks.
- **Vision/ViT**: ResNet/MobileNet for QAT/PTQ baselines; DeiT/ViT for transformer compression.

Tips:
- Store configs (YAML/JSON) and small metrics artifacts; keep checkpoints in releases or cloud buckets.
- Log experiment intent/results in a short `notes.md` so the README generator can link to it.
- Prefer container/venv definitions (e.g., `environment.yml`, `requirements.txt`) for repeatability.
