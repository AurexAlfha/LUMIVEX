import torch
import torch.nn as nn

from model import LumivexModel
from tokenizer import LumivexTokenizer


def load_validation_data(path="data/processed/validation.txt"):
    tokenizer = LumivexTokenizer()

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)

    return tokenizer, tokens


def evaluate(model, tokens, context_length):
    if len(tokens) < context_length + 1:
        raise ValueError(
            "Validation dataset is too small for the context length."
        )

    inputs = []
    targets = []

    for i in range(
        len(tokens) - context_length
    ):
        inputs.append(
            tokens[i:i + context_length]
        )
        targets.append(
            tokens[
                i + 1:i + context_length + 1
            ]
        )

    input_ids = torch.tensor(
        inputs,
        dtype=torch.long
    )

    target_ids = torch.tensor(
        targets,
        dtype=torch.long
    )

    model.eval()

    with torch.no_grad():
        output = model(input_ids)

        loss = nn.CrossEntropyLoss()(
            output.reshape(-1, output.size(-1)),
            target_ids.reshape(-1)
        )

    return loss.item()


if __name__ == "__main__":
    tokenizer, tokens = load_validation_data()

    model = LumivexModel()

    checkpoint = torch.load(
        "lumivex_v2_checkpoint.pt",
        map_location="cpu"
    )

    model.load_state_dict(checkpoint)

    loss = evaluate(
        model,
        tokens,
        model.context_length
    )

    print("LUMIVEX validation test successful.")
    print("Validation tokens:", len(tokens))
    print("Validation loss:", round(loss, 4))
