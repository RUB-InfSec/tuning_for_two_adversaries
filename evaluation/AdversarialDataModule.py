import lightning.pytorch as pl
import torch
import torchvision.transforms.v2 as transforms
from torch import Tensor
from torch.utils.data import DataLoader, Subset

import utils.utils
from training.CustomDataSet import SelfDataset


class AdversarialDataModule(pl.LightningDataModule):
    def __init__(self, attack_type: str = None, meta_data: str = None, file_name: str = None,
                 adversarial_images: Tensor = None, labels: Tensor = None):
        super().__init__()

        self.file_name = file_name
        self.attack_type = attack_type
        self.meta_data = meta_data

        self.adversarial_images = adversarial_images
        self.labels = labels

        self.batch_size = 128

        self.test_data = None

    def setup(self, stage=None):

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        transform = transforms.Compose([normalize])

        if self.attack_type != "square":
            data = torch.load(self.file_name)
            self.adversarial_images = data['images']
            self.labels = data['labels']

        self.test_data = SelfDataset(self.adversarial_images, self.labels, transform=transform)

    def test_dataloader(self):
        if utils.utils.is_debug_mode():
            return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        else:
            return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                              num_workers=8, pin_memory=True)

    def small_test_dataloader(self, size=1000):
        generator1 = torch.Generator().manual_seed(24)
        subset_indices = torch.randperm(len(self.test_data), generator=generator1)[
            :size]

        subset = Subset(self.test_data, subset_indices)

        return DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=0)
