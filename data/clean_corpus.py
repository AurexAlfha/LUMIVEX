from pathlib import Path


SOURCE = Path("data/processed/master_corpus.txt")
OUTPUT = Path("data/processed/clean_corpus.txt")


def clean_text(text):
    lines = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    unique_lines = []
    seen = set()

    for line in lines:
        key = line.casefold()

        if key not in seen:
            seen.add(key)
            unique_lines.append(line)

    return "\n".join(unique_lines)


text = SOURCE.read_text(encoding="utf-8")
cleaned = clean_text(text)

OUTPUT.write_text(
    cleaned,
    encoding="utf-8"
)

print("LUMIVEX CORPUS CLEANING SUCCESSFUL")
print("Original characters:", len(text))
print("Clean characters:", len(cleaned))
print("Original words:", len(text.split()))
print("Clean words:", len(cleaned.split()))
