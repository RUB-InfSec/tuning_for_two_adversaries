import lightning.pytorch as pl
import torch
from torch.utils.data import random_split, DataLoader, Subset

from training.ExperimentParameters import ExperimentParameters
from utils import utils


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.params: ExperimentParameters = ExperimentParameters.from_dict(config)

        self.train_data = None
        self.train_subset = None
        self.test_data = None
        self.val_data = None

    def setup(self, stage=None):
        self.train_data = self.params.get_dataset(train=True)
        self.test_data = self.params.get_dataset(train=False)

        test_abs = int(len(self.train_data) * (1.0 - self.params.data_parameters.validation_split))

        generator1 = torch.Generator().manual_seed(self.params.data_parameters.splitting_seed)
        self.train_subset, self.val_data = random_split(
            self.train_data, [test_abs, len(self.train_data) - test_abs], generator=generator1
        )

    def train_dataloader(self):
        if utils.is_debug_mode():
            return DataLoader(self.train_subset, batch_size=self.params.data_parameters.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
        else:
            return DataLoader(self.train_subset, batch_size=self.params.data_parameters.batch_size, shuffle=True,
                              num_workers=24,
                              pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.params.data_parameters.batch_size, shuffle=False,
                          num_workers=24,
                          pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        if utils.is_debug_mode():
            return DataLoader(self.test_data, batch_size=self.params.data_parameters.batch_size, shuffle=False,
                              num_workers=0)
        else:
            return DataLoader(self.test_data, batch_size=self.params.data_parameters.batch_size, shuffle=False,
                              num_workers=8, pin_memory=True)

    def small_train_dataloader(self, batches=1):
        generator1 = torch.Generator().manual_seed(self.params.data_parameters.splitting_seed)
        subset_indices = torch.randperm(len(self.train_subset), generator=generator1)[
            :batches * self.params.data_parameters.batch_size]

        subset = Subset(self.train_subset, subset_indices)

        if utils.is_debug_mode():
            return DataLoader(subset, batch_size=self.params.data_parameters.batch_size, shuffle=False, num_workers=0)
        else:
            return DataLoader(subset, batch_size=self.params.data_parameters.batch_size, shuffle=False, num_workers=8,
                              pin_memory=True)

    def small_test_dataloader(self, size=1000):
        generator1 = torch.Generator().manual_seed(self.params.data_parameters.splitting_seed)
        subset_indices = torch.randperm(len(self.test_data), generator=generator1)[
            :size]

        # Create a subset
        subset = Subset(self.test_data, subset_indices)

        if utils.is_debug_mode():
            return DataLoader(subset, batch_size=self.params.data_parameters.batch_size, shuffle=False, num_workers=0)
        else:
            return DataLoader(subset, batch_size=self.params.data_parameters.batch_size, shuffle=False, num_workers=8,
                              pin_memory=True)
