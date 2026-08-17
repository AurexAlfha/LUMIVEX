import torch
import torch.nn as nn
import torch.optim as optim

from model import LumivexModel
from training_pipeline import LumivexTrainingPipeline


print("Preparing LUMIVEX training...")

pipeline = LumivexTrainingPipeline()
inputs, targets = pipeline.prepare()

print("Training data ready.")
print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)

model = LumivexModel()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001
)

loss_function = nn.CrossEntropyLoss()

model.train()

epochs = 3

for epoch in range(epochs):
    optimizer.zero_grad()

    output = model(inputs)

    loss = loss_function(
        output.reshape(-1, output.size(-1)),
        targets.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    print(
        f"Epoch: {epoch + 1}/{epochs} "
        f"Loss: {loss.item():.4f}"
    )

torch.save(
    model.state_dict(),
    "lumivex_v2_checkpoint.pt"
)

print("LUMIVEX training completed.")
print("New checkpoint saved successfully.")
