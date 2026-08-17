from pathlib import Path

RAW_DIR = Path("data/raw")
EXTERNAL_DIR = Path("data/external/verified")
OUTPUT = Path("data/processed/master_corpus.txt")

sources = []

for path in sorted(RAW_DIR.rglob("*.txt")):
    sources.append(path)

for path in sorted(EXTERNAL_DIR.glob("*.txt")):
    sources.append(path)

texts = []

for path in sources:
    text = path.read_text(encoding="utf-8").strip()
    if text:
        texts.append(text)

master = "\n\n".join(texts)

OUTPUT.write_text(master, encoding="utf-8")

print("LUMIVEX MASTER CORPUS REBUILT")
print("Sources:", len(sources))
print("Characters:", len(master))
print("Words:", len(master.split()))
