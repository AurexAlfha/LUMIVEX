from pathlib import Path


class LumivexDatasetLoader:
    def __init__(self, root="data/raw"):
        self.root = Path(root)

    def files(self):
        return sorted(self.root.rglob("*.txt"))

    def load(self):
        documents = []

        for file in self.files():
            text = file.read_text(encoding="utf-8").strip()

            if text:
                documents.append({
                    "source": str(file),
                    "text": text
                })

        return documents


if __name__ == "__main__":
    loader = LumivexDatasetLoader()
    documents = loader.load()

    print("LUMIVEX DATASET LOADER READY")
    print("Documents:", len(documents))

    for item in documents:
        print("-", item["source"])
