import torch
import torch.nn as nn

from model import LumivexModel


torch.manual_seed(42)

model = LumivexModel(
    vocab_size=100,
    context_length=32
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

loss_function = nn.CrossEntropyLoss()

input_ids = torch.tensor([
    [10, 20, 30, 40, 50, 60, 70, 80],
    [20, 30, 40, 50, 60, 70, 80, 90],
    [30, 40, 50, 60, 70, 80, 90, 10],
], dtype=torch.long)

target_ids = torch.tensor([
    [20, 30, 40, 50, 60, 70, 80, 90],
    [30, 40, 50, 60, 70, 80, 90, 10],
    [40, 50, 60, 70, 80, 90, 10, 20],
], dtype=torch.long)

print("Starting LUMIVEX training...")

for step in range(20):
    optimizer.zero_grad()

    logits = model(input_ids)

    loss = loss_function(
        logits.reshape(-1, 100),
        target_ids.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        print("Step:", step, "Loss:", round(loss.item(), 4))

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": 20,
        "loss": loss.item(),
    },
    "lumivex_checkpoint.pt"
)

print("Training completed.")
print("Trained checkpoint saved successfully.")
