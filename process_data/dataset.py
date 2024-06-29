import torch
from torch.utils.data import Dataset, DataLoader


from ulti.read_data import read_tsv


class MIntRec(Dataset):
    def __init__(self, data: list) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        r"""
        {
            "text": [tokens, mask, segment_ids],
            "audio": torch.Tensor[batch_size, seq, hidden_dim],
            "video": torch.Tensor[batch_size, seq, hidden_dim],
            "label": int,
            "index": str,
        }
        """
        return self.data[idx]
