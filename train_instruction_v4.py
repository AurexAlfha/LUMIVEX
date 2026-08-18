import torch
import torch.nn as nn
import torch.optim as optim

from model import LumivexModel
from tokenizer import LumivexTokenizer


DATA = "data/processed/instruction_training.txt"
BASE = "lumivex_formatted_v3.pt"
OUTPUT = "lumivex_instruction_v4.pt"

tokenizer = LumivexTokenizer()

text = open(DATA, "r", encoding="utf-8").read()

examples = [
    block.strip()
    for block in text.split("\n\n")
    if block.strip()
]

model = LumivexModel()

state = torch.load(
    BASE,
    map_location="cpu"
)

model.load_state_dict(state)

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0003
)

loss_fn = nn.CrossEntropyLoss()

model.train()

max_length = model.context_length

for epoch in range(10):

    total_loss = 0.0
    count = 0

    for example in examples:

        response_marker = "<response>"

        if response_marker not in example:
            continue

        instruction, response = example.split(
            response_marker,
            1
        )

        instruction = instruction + response_marker
        response = response.strip()

        prompt_tokens = tokenizer.encode(instruction)
        response_tokens = tokenizer.encode(response)

        # Keep the complete response whenever possible.
        if len(response_tokens) >= max_length:
            response_tokens = response_tokens[:max_length - 1]

        available_prompt = max_length - len(response_tokens)

        if available_prompt < 2:
            continue

        prompt_tokens = prompt_tokens[:available_prompt]

        tokens = prompt_tokens + response_tokens

        if len(tokens) < 2:
            continue

        input_ids = torch.tensor(
            [tokens[:-1]],
            dtype=torch.long
        )

        target_ids = torch.tensor(
            [tokens[1:]],
            dtype=torch.long
        )

        optimizer.zero_grad()

        logits = model(input_ids)

        prompt_len = max(
            0,
            min(len(prompt_tokens) - 1, target_ids.size(1))
        )

        target = target_ids.clone()

        if prompt_len > 0:
            target[:, :prompt_len] = -100

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    average = total_loss / max(count, 1)

    print(
        f"V4 Epoch {epoch + 1}/10 | "
        f"Response Loss: {average:.4f}"
    )

torch.save(
    model.state_dict(),
    OUTPUT
)

print("LUMIVEX V4 INSTRUCTION TRAINING COMPLETE")
print("Checkpoint:", OUTPUT)
