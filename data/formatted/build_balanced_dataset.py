import json
from collections import defaultdict
from pathlib import Path

SOURCE = Path("data/formatted/example.jsonl")
OUTPUT = Path("data/formatted/balanced.jsonl")

categories = defaultdict(list)

for line in SOURCE.read_text(encoding="utf-8").splitlines():
    item = json.loads(line)

    category = item["category"].strip()
    text = " ".join(item["text"].split()).strip()

    if category and text:
        categories[category].append({
            "category": category,
            "text": text
        })

records = []

for category in sorted(categories):
    records.extend(categories[category])

OUTPUT.write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in records),
    encoding="utf-8"
)

print("LUMIVEX BALANCED DATASET BUILDER READY")
print("Categories:", len(categories))
print("Records:", len(records))

for category in sorted(categories):
    print(f"{category}: {len(categories[category])}")
