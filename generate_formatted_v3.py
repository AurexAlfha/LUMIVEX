import torch

from model import LumivexModel
from tokenizer import LumivexTokenizer


CHECKPOINT = "lumivex_formatted_v3.pt"

tokenizer = LumivexTokenizer()
model = LumivexModel()

state = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(state)
model.eval()

prompt = "LUMIVEX"

tokens = tokenizer.encode(prompt)

for _ in range(40):

    context = tokens[-model.context_length:]

    input_ids = torch.tensor(
        [context],
        dtype=torch.long
    )

    with torch.no_grad():
        logits = model(input_ids)

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1
    ).item()

    tokens.append(next_token)

output = tokenizer.decode(tokens)

print("LUMIVEX FORMATTED V3 GENERATION")
print("Prompt:", prompt)
print("Output:", output)
print("GENERATION SUCCESSFUL")
