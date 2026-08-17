from tokenizer import LumivexTokenizer

_tokenizer = LumivexTokenizer()

LUMIVEX_CONFIG = {
    "vocab_size": _tokenizer.vocab_size,
    "context_length": 128,
    "embedding_size": 256,
    "num_layers": 4,
    "num_heads": 4,
    "learning_rate": 0.001,
    "batch_size": 4,
}
