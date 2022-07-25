from parser import Namespace
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch_prac.data.dataset import CHDataset
from torch_prac.config import CHPath


class CHDataloader(CHPath):
    def __init__(self, config: Namespace, dataset: CHDataset):
        super().__init__(config)
        self.dataset = dataset
        self.dataloader: Optional[DataLoader] = None
        self.init_dataloader()

    def init_dataloader(self):
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def get_dataloader(self) -> DataLoader:
        return self.dataloader
