import csv
import torch
from torch.utils.data import Dataset
import nltk
from itertools import chain
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
import numpy as np
import pickle
import os
import sys

from data import benchmarks


class MMDataset(Dataset):
    def __init__(self, data):
        super(MMDataset, self).__init__()

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class MIntRec2:
    def __init__(self, tokenizer: str = "facebook/bart-base"):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.mapping = {
            "Acknowledge": "<<Acknowledge>>",
            "Advise": "<<Advise>>",
            "Agree": "<<Agree>>",
            "Apologise": "<<Apologise>>",
            "Arrange": "<<Arrange>>",
            "Ask for help": "<<Ask for help>>",
            "Asking for opinions": "<<Asking for opinions>>",
            "Care": "<<Care>>",
            "Comfort": "<<Comfort>>",
            "Complain": "<<Complain>>",
            "Confirm": "<<Confirm>>",
            "Criticize": "<<Criticize>>",
            "Doubt": "<<Doubt>>",
            "Emphasize": "<<Emphasize>>",
            "Explain": "<<Explain>>",
            "Flaunt": "<<Flaunt>>",
            "Greet": "<<Greet>>",
            "Inform": "<<Inform>>",
            "Introduce": "<<Introduce>>",
            "Invite": "<<Invite>>",
            "Joke": "<<Joke>>",
            "Leave": "<<Leave>>",
            "Oppose": "<<Oppose>>",
            "Plan": "<<Plan>>",
            "Praise": "<<Praise>>",
            "Prevent": "<<Prevent>>",
            "Refuse": "<<Refuse>>",
            "Taunt": "<<Taunt>>",
            "Thank": "<<Thank>>",
            "Warn": "<<Warn>>",
        }

        self.max_utt_num = 100
        add_tokens = list(self.mapping.values())
        add_tokens += ["<<U{}>>".format(i) for i in range(self.max_utt_num)]
        add_tokens += ["<<react>>", "<</react>>"]
        add_tokens += ["<<xReact{}>>".format(i) for i in range(self.max_utt_num)]
        add_tokens += ["<<oReact{}>>".format(i) for i in range(self.max_utt_num)]

        self.tokenizer.add_tokens(add_tokens)

        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= self.tokenizer.vocab_size
            self.mapping2id[key] = key_id[0]

        self.train_data: list = []
        self.dev_data: list = []
        self.test_data: list = []

        self.atomic_data: list = []
        self.kg_retrieval: list = []

    def tokenize_tokens(self, raw_words):
        w_bpes = [[self.tokenizer.bos_token_id]]
        for word in raw_words:
            bpes = self.tokenizer.tokenize(word)
            bpes = self.tokenizer.convert_tokens_to_ids(bpes)
            w_bpes.append(bpes)
        w_bpes.append([self.tokenizer.eos_token_id])
        all_bpes = list(chain(*w_bpes))
        return w_bpes, all_bpes

    @classmethod
    def get_prefix_ids(cls, word_bpes, start_prefix_id, last_prefix_id):
        utt_prefix_ids = []
        for ii, w_id in enumerate(word_bpes):
            if w_id >= start_prefix_id and w_id <= last_prefix_id:
                utt_prefix_ids.append(ii)
        return utt_prefix_ids

    def setup(self):
        self.kg_retrieval = MIntRec2.read_data("src/data/MintRec2/knowledge_retrieval.tsv")

        with open("src/data/MintRec2/atomic.csv", mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            self.atomic_data = [row for row in reader]

        self.dev_data = self.process_data("src/data/MintRec2/dev.tsv")
        self.test_data = self.process_data("src/data/MintRec2/test.tsv")
        self.train_data = self.process_data("src/data/MintRec2/train.tsv")

        # torch.save(MMDataset(self.dev_data), "src/data/dev_data.pth")
        # torch.save(MMDataset(self.test_data), "src/data/test_data.pth")
        # torch.save(MMDataset(self.train_data), "src/data/train_data.pth")

        self.tokenizer.save_pretrained("C:/Users/datng/Documents/LAB/KLTN/MintRec_code/shark/src/data/tokenizer")
        # torch.save(
        #     self.mapping2id,
        #     "C:/Users/datng/Documents/LAB/KLTN/MintRec_code/shark/src/data/mapping2id.pth",
        # )

        with open("src/data/dev_data.pkl", "wb") as f:
            pickle.dump(MMDataset(self.dev_data), f)

        with open("src/data/test_data.pkl", "wb") as f:
            pickle.dump(MMDataset(self.test_data), f)

        with open("src/data/train_data.pkl", "wb") as f:
            pickle.dump(MMDataset(self.train_data), f)

    @classmethod
    def read_data(cls, file_path: str):
        with open(file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader)
            return [row for row in reader]

    @classmethod
    def split_token(cls, word: str):
        return word_tokenize(word)

    @classmethod
    def clean_tokens(cls, tokens: list):
        return [token.replace("[", "").replace("]", "") for token in tokens]

    def process_data(self, file_path):
        data = MIntRec2.read_data(file_path)
        result: list = []

        speakers: list = []
        atomic = {"oReact": [], "xReact": []}
        retrieval = {"oReact": [], "xReact": []}
        word = []
        utt_prefix = {
            "ids": [],
            "atomic_xReact": [],
            "atomic_oReact": [],
            "retrieval_xReact": [],
            "retrieval_oReact": [],
        }

        for row in data:
            index = f"dia{row[0]}_utt{row[1]}"

            if row[1] == "0":
                atomic = {"oReact": [], "xReact": []}
                retrieval = {"oReact": [], "xReact": []}
                word = []
                speakers = []
                utt_prefix = {
                    "ids": [],
                    "atomic_xReact": [],
                    "atomic_oReact": [],
                    "retrieval_xReact": [],
                    "retrieval_oReact": [],
                }

            speakers.append(row[7])

            tmp_atomic = next((entry for entry in self.atomic_data if index in entry), None)

            atomic["oReact"] = (
                atomic["oReact"]
                + [f"<<oReact{row[1]}>>"]
                + MIntRec2.clean_tokens(MIntRec2.split_token(f"Others feel {tmp_atomic[2]}"))
            )
            atomic["xReact"] = (
                atomic["xReact"]
                + [f"<<xReact{row[1]}>>"]
                + MIntRec2.clean_tokens(MIntRec2.split_token(f"{row[7]} feels {tmp_atomic[8]}"))
            )

            tmp_retrieval = next((entry for entry in self.kg_retrieval if index in entry), None)

            retrieval["oReact"] = (
                retrieval["oReact"]
                + [f"<<oReact{row[1]}>>"]
                + MIntRec2.clean_tokens(MIntRec2.split_token(f"Others feel {tmp_retrieval[1]}"))
            )
            retrieval["xReact"] = (
                retrieval["xReact"]
                + [f"<<xReact{row[1]}>>"]
                + MIntRec2.clean_tokens(MIntRec2.split_token(f"{row[7]} feels {tmp_retrieval[2]}"))
            )

            word = word + [f"<<U{row[1]}>>"] + MIntRec2.clean_tokens(MIntRec2.split_token(f"{row[7]}: {row[2]}"))

            _word_bpes, word_bpes = self.tokenize_tokens(word)
            _, word_atomic_xReact = self.tokenize_tokens(atomic["xReact"])
            _, word_atomic_oReact = self.tokenize_tokens(atomic["oReact"])
            _, word_retrieval_xReact = self.tokenize_tokens(retrieval["xReact"])
            _, word_retrieval_oReact = self.tokenize_tokens(retrieval["oReact"])

            lens = list(map(len, _word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()  # [1,3,6,8]

            utt_prefix_ids = MIntRec2.get_prefix_ids(
                word_bpes,
                self.tokenizer.convert_tokens_to_ids("<<U0>>"),
                self.tokenizer.convert_tokens_to_ids(f"<<U{self.max_utt_num-1}>>"),
            )
            atomic_prefix_ids_xReact = MIntRec2.get_prefix_ids(
                word_atomic_xReact,
                self.tokenizer.convert_tokens_to_ids("<<xReact0>>"),
                self.tokenizer.convert_tokens_to_ids(f"<<xReact{self.max_utt_num-1}>>"),
            )
            atomic_prefix_ids_oReact = MIntRec2.get_prefix_ids(
                word_atomic_oReact,
                self.tokenizer.convert_tokens_to_ids("<<oReact0>>"),
                self.tokenizer.convert_tokens_to_ids(f"<<oReact{self.max_utt_num-1}>>"),
            )
            retrieval_prefix_ids_xReact = MIntRec2.get_prefix_ids(
                word_retrieval_xReact,
                self.tokenizer.convert_tokens_to_ids("<<xReact0>>"),
                self.tokenizer.convert_tokens_to_ids(f"<<xReact{self.max_utt_num-1}>>"),
            )
            retrieval_prefix_ids_oReact = MIntRec2.get_prefix_ids(
                word_retrieval_oReact,
                self.tokenizer.convert_tokens_to_ids("<<oReact0>>"),
                self.tokenizer.convert_tokens_to_ids(f"<<oReact{self.max_utt_num-1}>>"),
            )

            utt_xReact_mask = [[0] * self.max_utt_num for _ in range(self.max_utt_num)]
            utt_oReact_mask = [[0] * self.max_utt_num for _ in range(self.max_utt_num)]

            for ii in range(len(speakers)):
                start_id = 0
                for jj in np.arange(start_id, ii + 1):
                    if speakers[ii] == speakers[jj]:
                        utt_xReact_mask[ii][jj] = 1
                    else:
                        utt_oReact_mask[ii][jj] = 1

            if row[3] == "UNK":
                continue

            result.append(
                {
                    "index": index,
                    "dia_utt_num": int(row[1]),
                    "token": torch.LongTensor(word_bpes),
                    "len_token": len(word_bpes),
                    "word_atomic_oReact": torch.LongTensor(word_atomic_oReact),
                    "len_word_atomic_oReact": len(word_atomic_oReact),
                    "word_atomic_xReact": torch.LongTensor(word_atomic_xReact),
                    "len_word_atomic_xReact": len(word_atomic_xReact),
                    "word_retrieval_xReact": torch.LongTensor(word_retrieval_xReact),
                    "len_word_retrieval_xReact": len(word_retrieval_xReact),
                    "word_retrieval_oReact": torch.LongTensor(word_retrieval_oReact),
                    "len_word_retrieval_oReact": len(word_retrieval_oReact),
                    "utt_prefix_ids": torch.LongTensor(utt_prefix_ids),
                    "atomic_prefix_ids_xReact": torch.LongTensor(atomic_prefix_ids_xReact),
                    "atomic_prefix_ids_oReact": torch.LongTensor(atomic_prefix_ids_oReact),
                    "retrieval_prefix_ids_xReact": torch.LongTensor(retrieval_prefix_ids_xReact),
                    "retrieval_prefix_ids_oReact": torch.LongTensor(retrieval_prefix_ids_oReact),
                    "utt_xReact_mask": torch.LongTensor(utt_xReact_mask),
                    "utt_oReact_mask": torch.LongTensor(utt_oReact_mask),
                    "label": benchmarks["intent_labels"][row[3]],
                }
            )

        return result


MIntRec2().setup()
