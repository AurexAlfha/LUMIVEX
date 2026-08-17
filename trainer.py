import torch
import torch.nn as nn

from model import LumivexModel
from training_pipeline import LumivexTrainingPipeline


class LumivexTrainer:
    def __init__(self):
        self.pipeline = LumivexTrainingPipeline()

        self.model = LumivexModel(
            vocab_size=32000,
            context_length=128
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001
        )

        self.loss_function = nn.CrossEntropyLoss()

    def prepare_training(self):
        return self.pipeline.prepare()


if __name__ == "__main__":
    trainer = LumivexTrainer()
    print("LUMIVEX trainer configured.")
