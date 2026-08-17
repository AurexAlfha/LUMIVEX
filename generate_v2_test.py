import torch

from model import LumivexModel
from tokenizer import LumivexTokenizer

tokenizer = LumivexTokenizer()
model = LumivexModel()

checkpoint = torch.load(
    "lumivex_v2_checkpoint.pt",
    map_location="cpu"
)

model.load_state_dict(checkpoint)
model.eval()

prompt = "LUMIVEX"

tokens = tokenizer.encode(prompt)

with torch.no_grad():
    for _ in range(30):
        input_ids = torch.tensor(
            [tokens[-model.context_length:]],
            dtype=torch.long
        )

        logits = model(input_ids)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1
        ).item()

        tokens.append(next_token)

result = tokenizer.decode(tokens)

print("LUMIVEX V2 GENERATION SUCCESSFUL")
print("Prompt:", prompt)
print("Generated:", result)
