import threading
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from transformers import BertTokenizer
from typing import Any, Dict, Optional

from process_data.dataset import MIntRec
from ulti.read_data import read_tsv, read_pickle
from constants import Config


class MIntRecDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.data_train: Dataset
        self.data_val: Dataset
        self.data_test: Dataset

    def setup(self, stage: str):
        if stage == "fit":
            data_train = read_pickle(f"{Config.LOG_DATA_PATH}/data_train.pkl")
            data_val = read_pickle(f"{Config.LOG_DATA_PATH}/data_val.pkl")

            self.data_train = MIntRec(data_train)
            self.data_val = MIntRec(data_val)
        if stage == "test":
            data_test = read_pickle(f"{Config.LOG_DATA_PATH}/data_test.pkl")

            self.data_test = MIntRec(data_test)

        if stage == "predict":
            data_test = read_pickle(f"{Config.LOG_DATA_PATH}/data_test.pkl")

            self.data_test = MIntRec(data_test)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=Config.BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=Config.BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=Config.BATCH_SIZE)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=Config.BATCH_SIZE)
