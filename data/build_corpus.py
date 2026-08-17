from pathlib import Path


RAW_DIR = Path("data/raw")
EXTERNAL_DIR = Path("data/external/processed")
OUTPUT = Path("data/processed/master_corpus.txt")


sources = []


for path in sorted(RAW_DIR.rglob("*.txt")):
    sources.append(("core", path))


for path in sorted(EXTERNAL_DIR.glob("*.txt")):
    sources.append(("verified_external", path))


parts = []

for category, path in sources:

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        continue

    parts.append(
        f"[SOURCE: {category} | FILE: {path.name}]\n{text}"
    )


master = "\n\n".join(parts)

OUTPUT.write_text(
    master,
    encoding="utf-8"
)


print("LUMIVEX SOURCE-AWARE CORPUS CREATED")
print("Sources:", len(sources))
print("Characters:", len(master))
print("Words:", len(master.split()))
