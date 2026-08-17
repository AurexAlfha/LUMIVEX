from pathlib import Path

from tokenizer import LumivexTokenizer


class FormattedTrainingLoader:

    def __init__(
        self,
        path="data/processed/formatted_training.txt"
    ):
        self.path = Path(path)
        self.tokenizer = LumivexTokenizer()

    def load(self):
        text = self.path.read_text(
            encoding="utf-8"
        )

        tokens = self.tokenizer.encode(text)

        if not tokens:
            raise ValueError(
                "Formatted training dataset produced no tokens."
            )

        return tokens


if __name__ == "__main__":

    loader = FormattedTrainingLoader()
    tokens = loader.load()

    print("LUMIVEX FORMATTED TRAINING LOADER READY")
    print("Tokens:", len(tokens))
    print("Vocabulary:", loader.tokenizer.vocab_size)
