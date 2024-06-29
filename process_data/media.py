import torch
import numpy as np

from ulti.read_data import read_pickle


class MediaBase:
    def __init__(self, data_path: str, max_length: int, max_seq: int):
        self.data_path = data_path
        self.max_length = max_length
        self.max_seq = max_seq
        self.data: dict = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self) -> None:
        self.data = read_pickle(self.data_path)

    def get_value(self, key: str) -> torch.FloatTensor:
        return self.data[key].to(self.device)

    def _padding(self, feat: torch.FloatTensor) -> torch.FloatTensor:
        mdeia_length = feat.shape[0]
        if mdeia_length >= self.max_seq:
            return feat[: self.max_seq]

        pad = np.zeros([self.max_seq - mdeia_length, feat.shape[-1]])
        feat = np.concatenate((feat, pad), axis=0)

        return torch.FloatTensor(feat).to(self.device)

    def _padding_feats(self) -> dict:
        raise NotImplemented(f"Class {self.__class__} must implement")


class Audio(MediaBase):
    def __init__(self, data_path: str, max_length: int, max_seq: int):
        super().__init__(data_path, max_length, max_seq)

    def setup(self) -> None:
        super().setup()
        self.data = self._padding_feats()

    def _padding_feats(self) -> dict:
        padding_feats = {}

        for dataset_type in self.data.keys():
            feats = self.data[dataset_type]
            feats = np.array(feats)
            feats = self._padding(feats)

            padding_feats[dataset_type] = feats

        return padding_feats


class Video(MediaBase):
    def __init__(self, data_path: str, max_length: int, max_seq: int):
        super().__init__(data_path, max_length, max_seq)

    def setup(self) -> None:
        super().setup()
        self.data = self._padding_feats()

    def _padding_feats(self) -> dict:
        padding_feats = {}

        for dataset_type in self.data.keys():
            feats = self.data[dataset_type]
            feats = np.array(feats).squeeze(1)
            feats = self._padding(feats)

            padding_feats[dataset_type] = feats

        return padding_feats
