import os
import threading
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from transformers import BertTokenizer
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from process_data.dataset import MIntRec
from ulti.read_data import read_tsv, read_pickle

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)

DATA_DIR = os.getenv("DATA_DIR")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
LOG_DATA_PATH = os.getenv("LOG_DATA_PATH")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MIntRecDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.data_train: Dataset
        self.data_val: Dataset
        self.data_test: Dataset

    def setup(self, stage: str):
        if stage == "fit":
            data_train = read_pickle(f"{LOG_DATA_PATH}/data_train.pkl")
            data_val = read_pickle(f"{LOG_DATA_PATH}/data_val.pkl")

            self.data_train = MIntRec(data_train)
            self.data_val = MIntRec(data_val)
        if stage == "test":
            data_test = read_pickle(f"{LOG_DATA_PATH}/data_test.pkl")

            self.data_test = MIntRec(data_test)

        if stage == "predict":
            data_test = read_pickle(f"{LOG_DATA_PATH}/data_test.pkl")

            self.data_test = MIntRec(data_test)

    # def transfer_batch_to_device(self, batch, dataloader_idx: int, device: torch.device = device):
    #     pass

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=BATCH_SIZE)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=BATCH_SIZE)
