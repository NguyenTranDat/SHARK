import os
import pickle
import numpy as np

from constants import VideoConfig, ModelConfig

__all__ = ["VideoDataset"]


class VideoDataset:
    file_path = os.path.join(ModelConfig.PARENT_PATH, "data", ModelConfig.data_vesion, "video_feats.pkl")

    data = {}
    size = 0

    def __init__(self):
        with open(self.file_path, "rb") as file:
            self.data = pickle.load(file)

        self.padding_features()

    def __len__(self) -> int:
        return self.size

    def _padding(self, feat: np.array) -> np.array:
        video_length = feat.shape[0]
        if video_length >= VideoConfig.MAX_SEQ_LEN:
            return feat[: VideoConfig.MAX_SEQ_LEN]

        pad = np.zeros(
            [VideoConfig.MAX_SEQ_LEN - video_length, feat.shape[-1]]
        )
        feat = np.concatenate((feat, pad), axis=0)

        return feat

    def padding_features(self) -> None:
        padding_features = {}

        for key in self.data.keys():
            feats = self.data[key]
            feats = np.array(feats).squeeze(1)

            padding_features[key] = self._padding(feats)

        self.size = len(padding_features)

        self.data = padding_features

    def get_data(self, key: str) -> list:
        return self.data[key]
