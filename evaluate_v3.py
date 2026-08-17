import torch
import torch.nn as nn

from model import LumivexModel
from tokenizer import LumivexTokenizer


MODEL_PATH = "lumivex_v3_best.pt"
VALIDATION_PATH = "data/processed/master_validation.txt"


tokenizer = LumivexTokenizer()
model = LumivexModel()

state = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state)
model.eval()

text = open(
    VALIDATION_PATH,
    "r",
    encoding="utf-8"
).read()

tokens = tokenizer.encode(text)

context = model.context_length

if len(tokens) <= context:
    raise ValueError(
        f"Validation data too small: {len(tokens)} tokens"
    )

inputs = []
targets = []

for i in range(len(tokens) - context):
    inputs.append(tokens[i:i + context])
    targets.append(tokens[i + 1:i + context + 1])

x = torch.tensor(inputs, dtype=torch.long)
y = torch.tensor(targets, dtype=torch.long)

loss_fn = nn.CrossEntropyLoss()

with torch.no_grad():
    logits = model(x)

    loss = loss_fn(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )

perplexity = torch.exp(loss)

print("LUMIVEX V3 EVALUATION COMPLETE")
print("Validation tokens:", len(tokens))
print("Validation loss:", round(loss.item(), 6))
print("Perplexity:", round(perplexity.item(), 6))
