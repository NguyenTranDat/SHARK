from torch.utils.data import Dataset


class MMDataset(Dataset):
    def __init__(self, data):
        super(MMDataset, self).__init__()

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
