import torch
import pickle
from itertools import chain
from transformers import AutoTokenizer
import numpy as np
import csv
import shutil
import os

from src.example.mapping import benchmarks
from src.example.ulti import tokenize_raw_words, read_data, MAPPING, get_prefix_ids, clean_data


try:
    folder_path = "src/example/log"
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
except OSError as e:
    print(f"Không thể xoá thư mục {folder_path}: {e}")


class MIntRec2:
    def __init__(self, tokenizer: str = "facebook/bart-base"):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.max_utt_num = 100
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

    def tokenize_tokens(self, words):
        w_bpes = [[self.tokenizer.bos_token_id]]
        for word in words:
            bpes = self.tokenizer.tokenize(word)
            bpes = self.tokenizer.convert_tokens_to_ids(bpes)
            w_bpes.append(bpes)
        w_bpes.append([self.tokenizer.eos_token_id])
        all_bpes = list(chain(*w_bpes))
        return w_bpes, all_bpes

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

        with open("src/example/log/mapping2id.pkl", "wb") as f:
            pickle.dump(self.mapping2id, f)

    def process_data(self, file_path):
        data = read_data(file_path)
        result = []
        speakers, label, target_spans, words = [], [], [], []
        atomic, retrieval = {"oReact": [], "xReact": []}, {"oReact": [], "xReact": []}
        count = 0
        o_start_bpe, a_start_bpe = 0, 0

        for i, row in enumerate(data):
            index = f"dia{row[0]}_utt{row[1]}"

            if row[1] == "0":
                if count > 1:
                    result.append(
                        self._process_batch_data(index, words, atomic, retrieval, speakers, label, count, target_spans)
                    )
                speakers, label, target_spans, words = [], [], [], []
                atomic, retrieval = {"oReact": [], "xReact": []}, {"oReact": [], "xReact": []}
                count = 0

            if row[3] != "UNK":
                count += 1
                label.append(benchmarks["intent_labels"][row[3]])
                speakers.append(row[7])

                tmp_atomic = next((entry for entry in self.atomic_data if index in entry), None)
                atomic["oReact"] += tokenize_raw_words(f"<<oReact{count}>> Others feel {clean_data(tmp_atomic[2])}")
                atomic["xReact"] += tokenize_raw_words(f"<<xReact{count}>> {row[7]} feels {clean_data(tmp_atomic[8])}")

                tmp_retrieval = next((entry for entry in self.kg_retrieval if index in entry), None)
                retrieval["oReact"] += tokenize_raw_words(
                    f"<<oReact{count}>> Others feel {clean_data(tmp_retrieval[1])}"
                )
                retrieval["xReact"] += tokenize_raw_words(
                    f"<<xReact{count}>> {row[7]} feels {clean_data(tmp_retrieval[2])}"
                )

                if row[1] == "0":
                    o_start_bpe = 0
                    a_start_bpe = 0
                else:
                    o_start_bpe = a_start_bpe
                    a_start_bpe = len(words) + 1

                words += tokenize_raw_words(f"<<U{count}>> {row[7]}: {clean_data(row[2])}")

                target_spans.append([a_start_bpe + self.target_shift, o_start_bpe + self.target_shift])
                target_spans[-1].append(self.mapping2targetid[row[3]] + 2)
                target_spans[-1] = tuple(target_spans[-1])

        if words:
            result.append(
                self._process_batch_data(index, words, atomic, retrieval, speakers, label, count, target_spans)
            )

        return result

    def _process_batch_data(self, index, word, atomic, retrieval, speakers, label, count, target_spans):
        _, word_bpes = self.tokenize_tokens(word)
        _, word_atomic_xReact = self.tokenize_tokens(atomic["xReact"])
        _, word_atomic_oReact = self.tokenize_tokens(atomic["oReact"])
        _, word_retrieval_xReact = self.tokenize_tokens(retrieval["xReact"])
        _, word_retrieval_oReact = self.tokenize_tokens(retrieval["oReact"])

        target = [0]
        target.extend(list(chain(*target_spans)))
        target.append(1)

        utt_prefix_ids = get_prefix_ids(
            word_bpes,
            self.tokenizer.convert_tokens_to_ids("<<U0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<U{self.max_utt_num-1}>>"),
        )
        atomic_prefix_ids_xReact = get_prefix_ids(
            word_atomic_xReact,
            self.tokenizer.convert_tokens_to_ids("<<xReact0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<xReact{self.max_utt_num-1}>>"),
        )
        atomic_prefix_ids_oReact = get_prefix_ids(
            word_atomic_oReact,
            self.tokenizer.convert_tokens_to_ids("<<oReact0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<oReact{self.max_utt_num-1}>>"),
        )
        retrieval_prefix_ids_xReact = get_prefix_ids(
            word_retrieval_xReact,
            self.tokenizer.convert_tokens_to_ids("<<xReact0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<xReact{self.max_utt_num-1}>>"),
        )
        retrieval_prefix_ids_oReact = get_prefix_ids(
            word_retrieval_oReact,
            self.tokenizer.convert_tokens_to_ids("<<oReact0>>"),
            self.tokenizer.convert_tokens_to_ids(f"<<oReact{self.max_utt_num-1}>>"),
        )

        utt_xReact_mask = [[0] * self.max_utt_num for _ in range(self.max_utt_num)]
        utt_oReact_mask = [[0] * self.max_utt_num for _ in range(self.max_utt_num)]

        for ii in range(len(speakers)):
            start_id = 0
            for jj in range(start_id, ii + 1):
                if speakers[ii] == speakers[jj]:
                    utt_xReact_mask[ii][jj] = 1
                else:
                    utt_oReact_mask[ii][jj] = 1

        return {
            "index": index,
            "dia_utt_num": count,
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
            # "target": torch.LongTensor(target),
            # "len_target": len(target),
            "label": torch.LongTensor(label),
            # "len_label": len(label),
        }


MIntRec2().setup()
