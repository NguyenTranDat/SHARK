import os
import threading
import pickle
import torch
from torch.utils.data import Dataset
import lightning as L
from transformers import BertTokenizer
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from ulti.read_data import write_pickle, read_tsv
from process_data.benchmarks import benchmarks
from process_data.media import Audio, Video

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)

DATA_DIR = os.getenv("DATA_DIR")
DATA_VERSION = os.getenv("DATA_VERSION")
TOKENIZER = os.getenv("TOKENIZER")
LOG_DATA_PATH = os.getenv("LOG_DATA_PATH")

benchmark = benchmarks[DATA_VERSION]


class ProcessData:
    def __init__(self):
        self.data_train: Dataset
        self.data_val: Dataset
        self.data_test: Dataset

        self.audio = Audio(
            DATA_DIR + "audio_feats.pkl",
            max_length=benchmark["feat_dims"]["audio"],
            max_seq=benchmark["max_seq_lengths"]["audio"],
        )
        self.video = Video(
            DATA_DIR + "video_feats.pkl",
            max_length=benchmark["feat_dims"]["video"],
            max_seq=benchmark["max_seq_lengths"]["video"],
        )
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER, do_lower_case=True)

        self.label: dict = {}

        if DATA_VERSION in ["MintREC2.0"]:
            pass
        elif DATA_VERSION in ["MintREC"]:
            self.index_text = 3
            intent_labels = benchmark["intent_labels"]

        for intent_label in intent_labels:
            self.label[intent_label] = len(self.label)

    def setup(self):
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"The directory {DATA_DIR} does not exist.")
        # else:
        #     self.data_val = self.__make_data(DATA_DIR + "dev.tsv")
        #     self.data_train = self.__make_data(DATA_DIR + "train.tsv")
        #     self.data_test = self.__make_data(DATA_DIR + "test.tsv")

        self.audio.setup()
        self.video.setup()

        def load_data(attribute_name: str, file_path: str):
            setattr(self, attribute_name, self.__make_data(file_path))
            self.__save_data(attribute_name)

        threads = []
        threads.append(threading.Thread(target=load_data, args=("data_val", DATA_DIR + "dev.tsv")))
        threads.append(threading.Thread(target=load_data, args=("data_train", DATA_DIR + "train.tsv")))
        threads.append(threading.Thread(target=load_data, args=("data_test", DATA_DIR + "test.tsv")))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def __make_data(self, data_path: str):
        result = []
        data = read_tsv(data_path)
        max_seq_text = benchmark["max_seq_lengths"]["text"]

        for index, row in data.iterrows():
            text = row.iloc[self.index_text]
            label = self.label[row.iloc[self.index_text + 1]]

            index: str
            if DATA_VERSION in ["MintREC2.0"]:
                index = f"dia{row.iloc[0]}_utt{row.iloc[1]}"
            elif DATA_VERSION in ["MintREC"]:
                index = f"{row.iloc[0]}_{row.iloc[1]}_{row.iloc[2]}"

            audio = self.audio.get_value(index)
            video = self.video.get_value(index)

            token = self.tokenizer.tokenize(text)

            if len(token) > max_seq_text - 2:
                token = token[: (max_seq_text - 2)]

            tokens = ["[CLS]"] + token + ["[SEP]"]

            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (max_seq_text - len(tokens))
            mask = [1] * len(tokens) + padding
            segment_ids = [0] * len(tokens) + padding
            tokens += padding

            result.append(
                {
                    "text": torch.tensor([tokens, mask, segment_ids]),
                    "audio": torch.tensor(audio),
                    "video": torch.tensor(video),
                    "label": label,
                    "index": index,
                }
            )

        return result

    def __save_data(self, attribute_name):
        write_pickle(f"{LOG_DATA_PATH}/{attribute_name}.pkl", getattr(self, attribute_name, None))
