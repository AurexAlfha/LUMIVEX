from pathlib import Path

files = [
    Path("data/processed/final_training_corpus.txt"),
    Path("data/processed/master_train.txt"),
    Path("data/processed/master_validation.txt"),
]

print("LUMIVEX DATASET STATISTICS")
print("-" * 32)

for path in files:
    text = path.read_text(encoding="utf-8")
    print(f"{path.name}")
    print("  Characters:", len(text))
    print("  Words:", len(text.split()))
    print()

print("STATISTICS COMPLETE")
