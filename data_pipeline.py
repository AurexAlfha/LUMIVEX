from pathlib import Path

from tokenizer import LumivexTokenizer


class LumivexDataPipeline:
    def __init__(self, path="data/processed/master_train.txt"):
        self.path = Path(path)
        self.tokenizer = LumivexTokenizer()

    def encode_dataset(self):
        text = self.path.read_text(encoding="utf-8")
        return self.tokenizer.encode(text)


if __name__ == "__main__":
    pipeline = LumivexDataPipeline()
    tokens = pipeline.encode_dataset()

    print("LUMIVEX CLEAN TRAINING DATA READY")
    print("Tokens:", len(tokens))
    print("Vocabulary:", pipeline.tokenizer.vocab_size)
