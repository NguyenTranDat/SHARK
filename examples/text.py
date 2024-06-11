import pandas as pd
from transformers import BertTokenizer

from constants import ModelConfig, TextConfig

__all__ = [
    "TextFeatures",
    "TextDataset",
]


class TextFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class TextDataset:
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True,
    )

    def __init__(self, task: str, file_path: str):
        self.task = task
        self.file_path = file_path

        self.name_data = []
        self.text = []
        self.label = []

        self.features = []

        self.read_data()
        self.convert_examples_to_token()

        self.size = len(self.features)

    def __len__(self) -> int:
        return self.size

    def read_data(self) -> None:
        dataframe = pd.read_csv(self.file_path, sep="\t")

        if ModelConfig.data_vesion == "1.0":
            seasons = dataframe["season"]
            episodes = dataframe["episode"]
            clips = dataframe["clip"]
            self.name_data = [
                f"{seasons[i]}_{episodes[i]}_{clips[i]}"
                for i in range(len(seasons))
            ]
        elif ModelConfig.data_vesion == "2.0":
            dialogues = dataframe["Dialogue_id"]
            utterances = dataframe["Utterance_id"]
            self.name_data = [
                f"dia{dialogues[i]}_utt{utterances[i]}"
                for i in range(len(dialogues))
            ]

        self.text.extend(dataframe["text"])

        if self.task == ModelConfig.BINARY:
            self.label.extend(
                dataframe["label"].apply(ModelConfig.convert_label_binary)
            )
        else:
            self.label.extend(
                dataframe["label"].apply(ModelConfig.convert_label)
            )

    def convert_examples_to_token(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        for text in self.text:
            token = self.tokenizer.tokenize(text)

            if len(token) > TextConfig.MAX_SEQ_LEN - 2:
                token = token[: (TextConfig.MAX_SEQ_LEN - 2)]

            token = ["[CLS]"] + token + ["[SEP]"]

            input_ids = self.tokenizer.convert_tokens_to_ids(token)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(token)

            padding = [0] * (TextConfig.MAX_SEQ_LEN - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            self.features.append(
                TextFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                )
            )

    def get_features(self, idx: int) -> list:
        return self.features[idx]

    def get_label(self, idx: int) -> list:
        return self.label[idx]

    def get_name_data(self, idx: int) -> str:
        return self.name_data[idx]
