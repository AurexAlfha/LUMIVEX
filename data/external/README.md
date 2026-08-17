# LUMIVEX External Dataset Workspace

External datasets are stored separately from the core corpus.

Flow:

source
→ downloads
→ license verification
→ verified
→ cleaning
→ processed
→ master corpus

Rules:
- Record source and license before training.
- Keep raw source files unchanged.
- Never overwrite the original download.
- Do not use data whose terms prohibit the intended use.
- Start with small samples before scaling.
