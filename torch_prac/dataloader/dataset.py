import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CHDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 1:3].values
        text = row[0]
        y = row[1]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=15,
            pad_to_max_length=True,
            add_special_tokens=True,
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        return input_ids, attention_mask, y
