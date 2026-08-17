from dataset import LumivexTextDataset
from tokenizer import LumivexTokenizer


class LumivexDataPipeline:
    def __init__(self, path="data/sample.txt"):
        self.dataset = LumivexTextDataset(path)
        self.tokenizer = LumivexTokenizer()

    def encode_dataset(self):
        return self.tokenizer.encode(self.dataset.get_text())


if __name__ == "__main__":
    pipeline = LumivexDataPipeline()

    tokens = pipeline.encode_dataset()

    print("LUMIVEX data pipeline created.")
    print("Token count:", len(tokens))
