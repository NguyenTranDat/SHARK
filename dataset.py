import torch
import pickle
from itertools import chain
from transformers import AutoTokenizer
import numpy as np
import csv
import shutil
import os

from src.example.mapping import benchmarks
from src.example.ulti import read_data, MAPPING, get_prefix_ids, clean_data


try:
    folder_path = "src/example/log"
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
except OSError as e:
    print(f"Không thể xoá thư mục {folder_path}: {e}")


class MIntRec2:
    def __init__(self, tokenizer: str = "bert-base-cased"):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.max_utt_num = 65
        add_tokens = list(MAPPING.values())
        add_tokens += [f"<<U{i}>>" for i in range(self.max_utt_num)]
        add_tokens += [f"<<react>>", "<</react>>"]
        add_tokens += [f"<<xReact{i}>>" for i in range(self.max_utt_num)]
        add_tokens += [f"<<oReact{i}>>" for i in range(self.max_utt_num)]

        self.tokenizer.add_tokens(add_tokens)

        self.target_shift = len(MAPPING) + 1
        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in MAPPING.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

        self.train_data: list = []
        self.dev_data: list = []
        self.test_data: list = []

        self.atomic_data: list = []
        self.kg_retrieval: list = []

    def setup(self):
        self.kg_retrieval = read_data("src/example/MintRec2/knowledge_retrieval.tsv")

        with open("src/example/MintRec2/atomic.csv", mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            self.atomic_data = [row for row in reader]

        self.dev_data = self.process_data("src/example/MintRec2/dev.tsv")
        self.test_data = self.process_data("src/example/MintRec2/test.tsv")
        self.train_data = self.process_data("src/example/MintRec2/train.tsv")

        self.tokenizer.save_pretrained("src/example/log/tokenizer")

        with open("src/example/log/dev_data.pkl", "wb") as f:
            pickle.dump(self.dev_data, f)

        with open("src/example/log/test_data.pkl", "wb") as f:
            pickle.dump(self.test_data, f)

        with open("src/example/log/train_data.pkl", "wb") as f:
            pickle.dump(self.train_data, f)

        # with open("src/example/log/mapping2id.pkl", "wb") as f:
        # pickle.dump(self.mapping2id, f)

    def process_data(self, file_path):
        data = read_data(file_path)
        result = []
        speakers, label, words = [], [], []
        atomic, retrieval = {"oReact": [], "xReact": []}, {
            "oReact": [],
            "xReact": [],
        }
        count = 0

        for i, row in enumerate(data):
            if row[1] == "0":
                if count > 1:
                    result.append(self._process_batch_data(index, words, atomic, retrieval, speakers, label, count))
                speakers, label, words = [], [], [[], [], []]
                atomic, retrieval = {"oReact": [[], [], []], "xReact": [[], [], []]}, {
                    "oReact": [[], [], []],
                    "xReact": [[], [], []],
                }
                count = 0

            index = f"dia{row[0]}_utt{row[1]}"

            if row[3] != "UNK":
                label.append(benchmarks["intent_labels"][row[3]])
                speakers.append(row[7])

                tmp_atomic = next((entry for entry in self.atomic_data if index in entry), None)
                atomic_oReact_tokens = self.tokenizer.tokenize(
                    f"<<oReact{count}>> Others feel {clean_data(tmp_atomic[2])} [SEP]"
                )
                atomic_xReact_tokens = self.tokenizer.tokenize(
                    f"<<xReact{count}>> {row[7]} feels {clean_data(tmp_atomic[8])} [SEP]"
                )

                tmp_retrieval = next((entry for entry in self.kg_retrieval if index in entry), None)
                retrieval_oReact_tokens = self.tokenizer.tokenize(
                    f"<<oReact{count}>> Others feel {clean_data(tmp_retrieval[1])} [SEP]"
                )
                retrieval_xReact_tokens = self.tokenizer.tokenize(
                    f"<<xReact{count}>> {row[7]} feels {clean_data(tmp_retrieval[2])} [SEP]"
                )
                tokens = self.tokenizer.tokenize(f"<<U{count}>> {row[7]}: {clean_data(row[2])} [SEP]")

                if count == 0:
                    atomic_oReact_tokens = ["[CLS]"] + atomic_oReact_tokens
                    atomic_xReact_tokens = ["[CLS]"] + atomic_xReact_tokens
                    retrieval_oReact_tokens = ["[CLS]"] + retrieval_oReact_tokens
                    retrieval_xReact_tokens = ["[CLS]"] + retrieval_xReact_tokens
                    tokens = ["[CLS]"] + tokens

                atomic_oReact_ids = self.tokenizer.convert_tokens_to_ids(atomic_oReact_tokens)
                atomic_xReact_ids = self.tokenizer.convert_tokens_to_ids(atomic_xReact_tokens)
                retrieval_oReact_ids = self.tokenizer.convert_tokens_to_ids(retrieval_oReact_tokens)
                retrieval_xReact_ids = self.tokenizer.convert_tokens_to_ids(retrieval_xReact_tokens)
                tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                atomic["xReact"][0].extend(atomic_xReact_ids)
                atomic["xReact"][1].extend([count] * len(atomic_xReact_ids))
                atomic["xReact"][2].extend([0] * len(atomic_xReact_ids))

                atomic["oReact"][0].extend(atomic_oReact_ids)
                atomic["oReact"][1].extend([count] * len(atomic_oReact_ids))
                atomic["oReact"][2].extend([0] * len(atomic_oReact_ids))

                retrieval["xReact"][0].extend(retrieval_xReact_ids)
                retrieval["xReact"][1].extend([count] * len(retrieval_xReact_ids))
                retrieval["xReact"][2].extend([0] * len(retrieval_xReact_ids))

                retrieval["oReact"][0].extend(retrieval_oReact_ids)
                retrieval["oReact"][1].extend([count] * len(retrieval_oReact_ids))
                retrieval["oReact"][2].extend([0] * len(retrieval_oReact_ids))

                words[0].extend(tokens_ids)
                words[1].extend([count] * len(tokens_ids))
                words[2].extend([0] * len(tokens_ids))

                count += 1

        if words:
            result.append(self._process_batch_data(index, words, atomic, retrieval, speakers, label, count))

        return result

    def _process_batch_data(self, index, word, atomic, retrieval, speakers, label, count):
        utt_prefix_ids = get_prefix_ids(
            word[0],
            self.tokenizer.convert_tokens_to_ids("<<U0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<U{self.max_utt_num-1}>>"),
        )
        atomic_prefix_ids_xReact = get_prefix_ids(
            atomic["xReact"][0],
            self.tokenizer.convert_tokens_to_ids("<<xReact0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<xReact{self.max_utt_num-1}>>"),
        )
        atomic_prefix_ids_oReact = get_prefix_ids(
            atomic["oReact"][0],
            self.tokenizer.convert_tokens_to_ids("<<oReact0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<oReact{self.max_utt_num-1}>>"),
        )
        retrieval_prefix_ids_xReact = get_prefix_ids(
            retrieval["xReact"][0],
            self.tokenizer.convert_tokens_to_ids("<<xReact0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<xReact{self.max_utt_num-1}>>"),
        )
        retrieval_prefix_ids_oReact = get_prefix_ids(
            retrieval["oReact"][0],
            self.tokenizer.convert_tokens_to_ids("<<oReact0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<oReact{self.max_utt_num-1}>>"),
        )

        utt_xReact_mask = [[0] * self.max_utt_num for _ in range(self.max_utt_num)]
        utt_oReact_mask = [[0] * self.max_utt_num for _ in range(self.max_utt_num)]

        for ii in range(len(speakers)):
            for jj in range(ii + 1):
                if speakers[ii] == speakers[jj]:
                    utt_xReact_mask[ii][jj] = 1
                else:
                    utt_oReact_mask[ii][jj] = 1

        return {
            "index": index,
            "dia_utt_num": count,
            "token": torch.LongTensor(word),
            "len_token": len(word[0]),
            "word_atomic_oReact": torch.LongTensor(atomic["oReact"]),
            "len_word_atomic_oReact": len(atomic["oReact"][0]),
            "word_atomic_xReact": torch.LongTensor(atomic["xReact"]),
            "len_word_atomic_xReact": len(atomic["xReact"][0]),
            "word_retrieval_xReact": torch.LongTensor(retrieval["xReact"]),
            "len_word_retrieval_xReact": len(retrieval["oReact"][0]),
            "word_retrieval_oReact": torch.LongTensor(retrieval["oReact"]),
            "len_word_retrieval_oReact": len(retrieval["oReact"][0]),
            "utt_prefix_ids": torch.LongTensor(utt_prefix_ids),
            "atomic_prefix_ids_xReact": torch.LongTensor(atomic_prefix_ids_xReact),
            "atomic_prefix_ids_oReact": torch.LongTensor(atomic_prefix_ids_oReact),
            "retrieval_prefix_ids_xReact": torch.LongTensor(retrieval_prefix_ids_xReact),
            "retrieval_prefix_ids_oReact": torch.LongTensor(retrieval_prefix_ids_oReact),
            "utt_xReact_mask": torch.LongTensor(utt_xReact_mask),
            "utt_oReact_mask": torch.LongTensor(utt_oReact_mask),
            "label": torch.LongTensor(label),
        }


MIntRec2().setup()
