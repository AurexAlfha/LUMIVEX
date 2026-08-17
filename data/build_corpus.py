from pathlib import Path
from dataset_loader import LumivexDatasetLoader

output = Path("data/processed/master_corpus.txt")

loader = LumivexDatasetLoader("data/raw")
documents = loader.load()

if not documents:
    raise ValueError("No training documents found.")

master_text = "\n\n".join(
    item["text"] for item in documents
).strip()

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(master_text, encoding="utf-8")

print("LUMIVEX master corpus rebuilt successfully.")
print("Documents:", len(documents))
print("Characters:", len(master_text))
print("Words:", len(master_text.split()))
