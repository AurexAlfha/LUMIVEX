import json
from pathlib import Path

SOURCE = Path("data/formatted/balanced.jsonl")
OUTPUT = Path("data/processed/formatted_training.txt")

records = []

for line in SOURCE.read_text(encoding="utf-8").splitlines():

    item = json.loads(line)

    category = item["category"].strip().lower()
    text = " ".join(item["text"].split()).strip()

    if not category or not text:
        continue

    records.append(
        f"<category>{category}</category>\n{text}"
    )

if not records:
    raise ValueError("No valid training records found.")

OUTPUT.write_text(
    "\n\n".join(records),
    encoding="utf-8"
)

print("LUMIVEX FORMATTED TRAINING TEXT READY")
print("Records:", len(records))
print("Words:", len(OUTPUT.read_text(encoding="utf-8").split()))
