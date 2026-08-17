import torch
import torch.nn as nn
import torch.optim as optim

from model import LumivexModel
from tokenizer import LumivexTokenizer
from config import LUMIVEX_CONFIG


TRAIN_PATH = "data/processed/master_train.txt"
VALIDATION_PATH = "data/processed/master_validation.txt"


def load_tokens(path):
    tokenizer = LumivexTokenizer()

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return tokenizer.encode(text)


def make_sequences(tokens, context_length):
    if len(tokens) <= context_length:
        raise ValueError(
            f"Not enough tokens: {len(tokens)} "
            f"for context length {context_length}"
        )

    inputs = []
    targets = []

    for i in range(len(tokens) - context_length):
        inputs.append(
            tokens[i:i + context_length]
        )
        targets.append(
            tokens[i + 1:i + context_length + 1]
        )

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


print("Preparing LUMIVEX V3 training...")

context_length = LUMIVEX_CONFIG["context_length"]

train_tokens = load_tokens(TRAIN_PATH)
validation_tokens = load_tokens(VALIDATION_PATH)

train_x, train_y = make_sequences(
    train_tokens,
    context_length
)

val_x, val_y = make_sequences(
    validation_tokens,
    context_length
)

print("Training tokens:", len(train_tokens))
print("Validation tokens:", len(validation_tokens))
print("Training sequences:", len(train_x))
print("Validation sequences:", len(val_x))

model = LumivexModel()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LUMIVEX_CONFIG["learning_rate"]
)

loss_function = nn.CrossEntropyLoss()

batch_size = 4
epochs = 3

best_validation_loss = float("inf")

for epoch in range(epochs):

    model.train()

    total_train_loss = 0.0
    train_batches = 0

    for start in range(0, len(train_x), batch_size):

        end = min(
            start + batch_size,
            len(train_x)
        )

        inputs = train_x[start:end]
        targets = train_y[start:end]

        optimizer.zero_grad()

        output = model(inputs)

        loss = loss_function(
            output.reshape(-1, output.size(-1)),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_batches += 1

    average_train_loss = (
        total_train_loss / train_batches
    )

    model.eval()

    with torch.no_grad():

        val_output = model(val_x)

        validation_loss = loss_function(
            val_output.reshape(-1, val_output.size(-1)),
            val_y.reshape(-1)
        ).item()

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Validation Loss: {validation_loss:.4f}"
    )

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            model.state_dict(),
            "lumivex_v3_best.pt"
        )

        print("Best V3 checkpoint saved.")

print("LUMIVEX V3 training completed.")
