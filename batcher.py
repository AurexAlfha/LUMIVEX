import torch


class LumivexBatcher:
    def __init__(self, context_length=128):
        self.context_length = context_length

    def create_batch(self, tokens):
        if len(tokens) < self.context_length + 1:
            raise ValueError(
                "Not enough tokens for a training sequence."
            )

        inputs = []
        targets = []

        for i in range(
            len(tokens) - self.context_length
        ):
            inputs.append(
                tokens[i:i + self.context_length]
            )

            targets.append(
                tokens[
                    i + 1:i + self.context_length + 1
                ]
            )

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )
