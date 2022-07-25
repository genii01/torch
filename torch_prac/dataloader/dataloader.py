from argparse import Namespace
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch_prac.dataloader.dataset import CHDataset
from torch_prac.config import CHPath


class CHDataloader(CHPath):
    def __init__(self, config: Namespace, dataset: CHDataset, tokenizer):
        super().__init__(config)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.dataloader: Optional[DataLoader] = None
        self.init_dataloader()

    def init_dataloader(self):
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.custom_collate_fn,
            shuffle=True,
        )

    def custom_collate_fn(self, batch_samples):
        # labels 리스트 값
        labels = [sample["label"] for sample in batch_samples]
        # texts 리스트 값
        texts = [sample["text"] for sample in batch_samples]

        encoded_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
            "labels": torch.tensor(labels),
        }

    def get_dataloader(self) -> DataLoader:
        return self.dataloader
