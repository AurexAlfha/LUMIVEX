from config import LUMIVEX_CONFIG
from data_pipeline import LumivexDataPipeline
from batcher import LumivexBatcher


class LumivexTrainingPipeline:
    def __init__(self):
        self.config = LUMIVEX_CONFIG
        self.data = LumivexDataPipeline()

        self.batcher = LumivexBatcher(
            context_length=self.config["context_length"]
        )

    def prepare(self):
        tokens = self.data.encode_dataset()

        return self.batcher.create_batch(tokens)
