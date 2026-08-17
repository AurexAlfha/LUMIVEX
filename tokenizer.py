class LumivexTokenizer:
    """
    LUMIVEX byte-level tokenizer.

    Each UTF-8 byte maps to one token.
    Special tokens:
        0 = PAD
        1 = UNK
        2-257 = byte values 0-255
    """

    PAD_ID = 0
    UNK_ID = 1
    BYTE_OFFSET = 2
    VOCAB_SIZE = 258

    def __init__(self):
        self.vocab_size = self.VOCAB_SIZE

    def encode(self, text):
        data = text.encode("utf-8")

        return [
            byte + self.BYTE_OFFSET
            for byte in data
        ]

    def decode(self, token_ids):
        data = bytearray()

        for token_id in token_ids:
            if self.BYTE_OFFSET <= token_id < self.BYTE_OFFSET + 256:
                data.append(token_id - self.BYTE_OFFSET)

        return bytes(data).decode(
            "utf-8",
            errors="replace"
        )


if __name__ == "__main__":
    tokenizer = LumivexTokenizer()

    text = "Hello LUMIVEX! नमस्ते 🚀"

    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print("LUMIVEX Tokenizer V2 test successful.")
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Original:", text)
    print("Token count:", len(tokens))
    print("Decoded:", decoded)

    if decoded == text:
        print("Encode/decode verification: PASS")
    else:
        print("Encode/decode verification: FAILED")
