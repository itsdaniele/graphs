from typing import Optional
from torch_geometric.datasets import PPI

from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader, Dataset


class PYGDatamodule(LightningDataModule):
    def __init__(
        self,
        dataset: str = "ppi",
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.dataset = dataset
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform = None
        self.pre_transform = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if not self.data_train and not self.data_val and not self.data_test:

            if self.dataset == "ppi":
                self.data_train = PPI(
                    self.data_dir, split="train", pre_transform=self.pre_transform
                )
                self.data_val = PPI(
                    self.data_dir, split="val", pre_transform=self.pre_transform
                )
                self.data_test = PPI(
                    self.data_dir, split="test", pre_transform=self.pre_transform
                )
            else:
                raise NotImplementedError(f"Dataset {self.dataset} not supported")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
