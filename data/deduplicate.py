from pathlib import Path


SOURCE = Path("data/processed/clean_corpus.txt")
OUTPUT = Path("data/processed/deduplicated_corpus.txt")


text = SOURCE.read_text(encoding="utf-8")

lines = []
seen = set()

for line in text.splitlines():

    line = " ".join(line.split()).strip()

    if not line:
        continue

    key = line.casefold()

    if key not in seen:
        seen.add(key)
        lines.append(line)


result = "\n".join(lines)

OUTPUT.write_text(
    result,
    encoding="utf-8"
)

print("LUMIVEX DEDUPLICATION COMPLETE")
print("Original lines:", len(text.splitlines()))
print("Unique lines:", len(lines))
print("Removed:", len(text.splitlines()) - len(lines))
