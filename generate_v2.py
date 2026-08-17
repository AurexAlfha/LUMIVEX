import torch

from model import LumivexModel
from tokenizer import LumivexTokenizer


tokenizer = LumivexTokenizer()
model = LumivexModel()

state = torch.load(
    "lumivex_v2_checkpoint.pt",
    map_location="cpu"
)

model.load_state_dict(state)
model.eval()


def generate(prompt, max_new_tokens=30):
    tokens = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        input_ids = torch.tensor(
            [tokens[-model.context_length:]],
            dtype=torch.long
        )

        with torch.no_grad():
            logits = model(input_ids)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1
        ).item()

        tokens.append(next_token)

    return tokenizer.decode(tokens)


prompt = "LUMIVEX"

print("LUMIVEX V2 generation test started.")
print("Prompt:", prompt)

result = generate(prompt)

print("Generated text:")
print(result)

print("LUMIVEX V2 generation test completed.")
