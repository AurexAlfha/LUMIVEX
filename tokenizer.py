class LumivexTokenizer:
    def __init__(self):
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

    def encode(self, text):
        words = text.strip().split()

        tokens = [self.special_tokens["<BOS>"]]

        for word in words:
            token_id = sum(ord(char) for char in word) % 96
            tokens.append(token_id + 4)

        tokens.append(self.special_tokens["<EOS>"])
        return tokens

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


if __name__ == "__main__":
    tokenizer = LumivexTokenizer()

    text = "Hello LUMIVEX"
    tokens = tokenizer.encode(text)

    print("LUMIVEX tokenizer loaded successfully.")
    print("Text:", text)
    print("Tokens:", tokens)
    print("Decoded:", tokenizer.decode(tokens))
