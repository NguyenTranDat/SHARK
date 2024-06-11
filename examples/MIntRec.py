from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ["MIntRecDataset"]


class MIntRecDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.size = len(data)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        text = []
        text.append(self.data[index]["text"].input_ids)
        text.append(self.data[index]["text"].input_mask)
        text.append(self.data[index]["text"].segment_ids)
        text = torch.tensor(text)

        video = torch.tensor(np.array(self.data[index]["video"]))
        audio = torch.tensor(np.array(self.data[index]["audio"]))

        sample = {
            "label": self.data[index]["label"],
            "text": torch.tensor(text),
            "video": video,
            "audio": audio,
        }
        return sample
