import torch

from model import LumivexModel
from tokenizer import LumivexTokenizer


CHECKPOINT = "lumivex_instruction_v4.pt"

tokenizer = LumivexTokenizer()
model = LumivexModel()

state = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(state)
model.eval()

print("LUMIVEX CHAT")
print("Type 'exit' to stop.")
print("-" * 30)

while True:

    prompt = input("You: ").strip()

    if prompt.lower() == "exit":
        print("LUMIVEX: Goodbye.")
        break

    if not prompt:
        continue

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

    response = tokenizer.decode(tokens)

    print("LUMIVEX:", response)
    print()
