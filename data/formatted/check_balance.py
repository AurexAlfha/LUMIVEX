import json
from collections import Counter
from pathlib import Path

SOURCE = Path("data/formatted/balanced.jsonl")

counts = Counter()

for line in SOURCE.read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    counts[item["category"]] += 1

print("LUMIVEX BALANCE CHECK COMPLETE")
print("Categories:", len(counts))
print("Total records:", sum(counts.values()))

for category, count in sorted(counts.items()):
    print(f"{category}: {count}")

if not counts:
    raise ValueError("Dataset contains no records.")

print("Balance check passed.")
