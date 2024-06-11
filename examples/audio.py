import os
import pickle
import numpy as np

from constants import AudioConfig, ModelConfig

__all__ = ["AudioDataset"]


class AudioDataset:
    file_path = os.path.join(ModelConfig.PARENT_PATH, "data", ModelConfig.data_vesion, "audio_feats.pkl")

    data = {}
    size = 0

    def __init__(self):
        with open(self.file_path, "rb") as file:
            self.data = pickle.load(file)

        self.padding_features()

    def __len__(self) -> int:
        return self.size

    def __padding(self, feat: np.array) -> np.array:
        audio_length = feat.shape[0]
        if audio_length >= AudioConfig.MAX_SEQ_LEN:
            return feat[: AudioConfig.MAX_SEQ_LEN]

        pad = np.zeros(
            [AudioConfig.MAX_SEQ_LEN - audio_length, feat.shape[-1]]
        )
        feat = np.concatenate((feat, pad), axis=0)

        return feat

    def padding_features(self) -> None:
        padding_features = {}

        for key, value in self.data.items():
            value = np.array(value)
            padding_features[key] = self.__padding(value)

        self.size = len(padding_features)

        self.data = padding_features

    def get_data(self, key: str) -> list:
        return self.data[key]
