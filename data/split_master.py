from pathlib import Path


source = Path("data/processed/master_corpus.txt")

train_file = Path("data/processed/master_train.txt")
validation_file = Path("data/processed/master_validation.txt")

text = source.read_text(encoding="utf-8").strip()

words = text.split()

if len(words) < 10:
    raise ValueError("Master corpus is too small to split.")

split_index = int(len(words) * 0.8)

train_text = " ".join(words[:split_index])
validation_text = " ".join(words[split_index:])

train_file.write_text(
    train_text,
    encoding="utf-8"
)

validation_file.write_text(
    validation_text,
    encoding="utf-8"
)

print("LUMIVEX master dataset split successfully.")
print("Training words:", len(train_text.split()))
print("Validation words:", len(validation_text.split()))
