import csv
import re


MAPPING = {
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


def read_data(file_path: str):
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)
        return [row for row in reader]


def get_prefix_ids(word_bpes, start_prefix_id, last_prefix_id):
    utt_prefix_ids = []
    for ii, w_id in enumerate(word_bpes):
        if w_id >= start_prefix_id and w_id <= last_prefix_id:
            utt_prefix_ids.append(ii)
    return utt_prefix_ids


def clean_data(data):
    pattern = r"[^a-zA-Z0-9]"
    cleaned_data = re.sub(pattern, " ", data)
    return cleaned_data


def tokenize_raw_words(text):
    tokens = []
    pattern = r"<<xReact\d+>>|<<oReact\d+>>|<<U\d+>>|[a-zA-Z0-9'-]+|[\.,!?;:]"
    matches = re.findall(pattern, text)
    tokens.extend(matches)

    return tokens
