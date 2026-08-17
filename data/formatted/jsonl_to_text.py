import json
from pathlib import Path

SOURCE = Path("data/formatted/example.jsonl")
OUTPUT = Path("data/formatted/example_converted.txt")

lines = []

for line in SOURCE.read_text(encoding="utf-8").splitlines():

    item = json.loads(line)

    category = item["category"].strip()
    text = " ".join(item["text"].split()).strip()

    if text:
        lines.append(
            f"[CATEGORY: {category}]\n{text}"
        )

OUTPUT.write_text(
    "\n\n".join(lines),
    encoding="utf-8"
)

print("LUMIVEX JSONL CONVERSION SUCCESSFUL")
print("Records:", len(lines))
print("Words:", len(OUTPUT.read_text(encoding="utf-8").split()))
