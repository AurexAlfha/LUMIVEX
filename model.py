import torch
import torch.nn as nn

from config import LUMIVEX_CONFIG


class LumivexModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        config = config or LUMIVEX_CONFIG

        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        self.embedding_size = config["embedding_size"]

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_size
        )

        self.position_embedding = nn.Embedding(
            self.context_length,
            self.embedding_size
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=config["num_heads"],
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config["num_layers"]
        )

        self.output = nn.Linear(
            self.embedding_size,
            self.vocab_size
        )

    def forward(self, input_ids):
        length = input_ids.shape[1]

        if length > self.context_length:
            raise ValueError("Input exceeds LUMIVEX context length.")

        positions = torch.arange(
            length,
            device=input_ids.device
        )

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)

        x = self.transformer(x)

        return self.output(x)


if __name__ == "__main__":
    model = LumivexModel()

    test_input = torch.randint(
        0,
        model.vocab_size,
        (1, 16)
    )

    output = model(test_input)

    print("LUMIVEX model configuration test successful.")
    print("Vocabulary:", model.vocab_size)
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)
