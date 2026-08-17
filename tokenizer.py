class LumivexTokenizer:
    def __init__(self):
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
        }

        for i, char in enumerate(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?'-\n",
            start=2
        ):
            if char not in self.vocab:
                self.vocab[char] = i

        self.inverse_vocab = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        return [
            self.vocab.get(char, self.vocab["<UNK>"])
            for char in text
        ]

    def decode(self, token_ids):
        return "".join(
            self.inverse_vocab.get(token_id, "<UNK>")
            for token_id in token_ids
        )


if __name__ == "__main__":
    tokenizer = LumivexTokenizer()

    text = "Hello LUMIVEX!"

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print("LUMIVEX tokenizer test successful.")
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Text:", text)
    print("Tokens:", tokens)
    print("Decoded:", decoded)
