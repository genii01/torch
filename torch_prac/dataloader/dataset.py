import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CHDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 1:3].values
        text = row[0]
        label = row[1]
        return {"text": text, "label": label}
