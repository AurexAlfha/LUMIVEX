import torch
import torch.nn as nn
import torch.optim as optim

from model import LumivexModel
from training_pipeline import LumivexTrainingPipeline

print("Preparing LUMIVEX memory-efficient training...")

pipeline = LumivexTrainingPipeline()
inputs, targets = pipeline.prepare()

print("Training data ready.")
print("Total sequences:", inputs.shape[0])
print("Context length:", inputs.shape[1])

model = LumivexModel()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001
)

loss_function = nn.CrossEntropyLoss()

model.train()

batch_size = 4
max_batches = 50

total_batches = min(
    (len(inputs) + batch_size - 1) // batch_size,
    max_batches
)

for batch_number in range(total_batches):
    start = batch_number * batch_size
    end = min(start + batch_size, len(inputs))

    batch_inputs = inputs[start:end]
    batch_targets = targets[start:end]

    optimizer.zero_grad()

    output = model(batch_inputs)

    loss = loss_function(
        output.reshape(-1, output.size(-1)),
        batch_targets.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    if batch_number % 5 == 0:
        print(
            f"Batch: {batch_number + 1}/{total_batches} "
            f"Loss: {loss.item():.4f}"
        )

torch.save(
    model.state_dict(),
    "lumivex_v2_checkpoint.pt"
)

print("LUMIVEX memory-efficient training completed.")
print("New checkpoint saved successfully.")
