from pathlib import Path

SOURCE = Path("data/processed/deduplicated_corpus.txt")
OUTPUT = Path("data/processed/final_training_corpus.txt")

text = SOURCE.read_text(encoding="utf-8").strip()

if len(text.split()) < 20:
    raise ValueError("Training corpus is too small.")

OUTPUT.write_text(text, encoding="utf-8")

print("LUMIVEX FINAL TRAINING CORPUS READY")
print("Words:", len(text.split()))
print("Characters:", len(text))
