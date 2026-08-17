from pathlib import Path


source = Path("data/raw/lumivex_corpus.txt")
output = Path("data/processed/lumivex_corpus.txt")

text = source.read_text(encoding="utf-8")

lines = [
    " ".join(line.split())
    for line in text.splitlines()
    if line.strip()
]

clean_text = "\n".join(lines)

output.write_text(
    clean_text,
    encoding="utf-8"
)

print("LUMIVEX corpus prepared successfully.")
print("Characters:", len(clean_text))
print("Lines:", len(lines))
