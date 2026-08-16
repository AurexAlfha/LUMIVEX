import torch
import torch.nn as nn

class LumivexModel(nn.Module):
    def __init__(self, vocab_size=32000, context_length=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 256)
        self.position_embedding = nn.Embedding(context_length, 256)

        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.output = nn.Linear(256, vocab_size)

    def forward(self, input_ids):
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)
        x = self.transformer(x)

        return self.output(x)

if __name__ == "__main__":
    model = LumivexModel()
    test_input = torch.randint(0, 32000, (1, 16))
    output = model(test_input)

    print("LUMIVEX Transformer loaded successfully.")
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)
