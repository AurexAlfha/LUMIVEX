from pathlib import Path


source = Path("data/processed/lumivex_corpus.txt")
train_file = Path("data/processed/train.txt")
validation_file = Path("data/processed/validation.txt")

text = source.read_text(encoding="utf-8")

lines = [
    line.strip()
    for line in text.splitlines()
    if line.strip()
]

split_index = max(1, int(len(lines) * 0.8))

train_lines = lines[:split_index]
validation_lines = lines[split_index:]

train_file.write_text(
    "\n".join(train_lines),
    encoding="utf-8"
)

validation_file.write_text(
    "\n".join(validation_lines),
    encoding="utf-8"
)

print("LUMIVEX dataset split successfully.")
print("Training lines:", len(train_lines))
print("Validation lines:", len(validation_lines))
