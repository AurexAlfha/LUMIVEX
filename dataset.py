from pathlib import Path


class LumivexTextDataset:
    def __init__(self, path="data/sample.txt"):
        self.path = Path(path)
        self.text = self.path.read_text(encoding="utf-8")

    def __len__(self):
        return len(self.text)

    def get_text(self):
        return self.text


if __name__ == "__main__":
    dataset = LumivexTextDataset()

    print("LUMIVEX dataset loaded successfully.")
    print("Characters:", len(dataset))
    print("Preview:", dataset.get_text()[:80])
