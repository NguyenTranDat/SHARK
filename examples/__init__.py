import os
import torch
from torch.utils.data import DataLoader

from .text import TextDataset
from .audio import AudioDataset
from .video import VideoDataset
from .MIntRec import MIntRecDataset

from constants import ModelConfig


class DataMIntRec:
    path_data = None

    audio = AudioDataset()
    video = VideoDataset()

    def __init__(self):
        self.path_data = os.path.join(ModelConfig.PARENT_PATH, "data", ModelConfig.data_vesion, "test.tsv")
        self.data_test = MIntRecDataset(self.setup_data())

        self.path_data = os.path.join(ModelConfig.PARENT_PATH, "data", ModelConfig.data_vesion, "dev.tsv")
        self.data_dev = MIntRecDataset(self.setup_data())

        self.path_data = os.path.join(ModelConfig.PARENT_PATH, "data", ModelConfig.data_vesion, "train.tsv")
        self.data_train = MIntRecDataset(self.setup_data())

    def setup_data(self, augment: bool = False) -> list:
        data = []
        text = TextDataset(task=ModelConfig.task, file_path=self.path_data)

        for index in range(text.size):
            output = {}
            output["text"] = text.get_features(index)
            output["label"] = text.get_label(index)

            if not augment:
                name_data = text.get_name_data(index)
                output["audio"] = self.audio.get_data(name_data)
                output["video"] = self.video.get_data(name_data)

            data.append(output)

        del text

        return data

    def get_dataloader_augment(self):
        self.path_data = os.path.join(ModelConfig.PARENT_PATH, "data", "augment_text_sdif.tsv")
        self.data_augment = MIntRecDataset(self.setup_data(augment=True))
        aug_dataloader = DataLoader(
            self.data_augment, batch_size=ModelConfig.BATCH_SIZE
        )
        return aug_dataloader

    def get_dataloader(self):
        train_dataloader = DataLoader(
            self.data_train, batch_size=ModelConfig.BATCH_SIZE
        )

        test_dataloader = DataLoader(
            self.data_test, batch_size=ModelConfig.BATCH_SIZE
        )

        dev_dataloader = DataLoader(
            self.data_dev, batch_size=ModelConfig.BATCH_SIZE
        )

        return (train_dataloader, dev_dataloader, test_dataloader)
