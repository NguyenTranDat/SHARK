import pandas as pd
import pickle


def read_tsv(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path, sep="\t")

    data = data.iloc[1:].reset_index(drop=True)

    return data


def read_pickle(file_path: str):
    data = pd.read_pickle(file_path)

    return data


def write_pickle(file_path: str, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
