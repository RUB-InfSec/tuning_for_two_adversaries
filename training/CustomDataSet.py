import torch
from torch.utils.data import Dataset


class SplitByIndexDataset(Dataset):
    def __init__(self, idx, dataset):
        self.idx = idx
        self.ds = dataset

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        realid = self.idx[idx]
        img, label = self.ds[realid]
        return img, label


class SelfDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.t = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.t is None:
            return self.images[index], self.labels[index]
        else:
            return self.t(self.images[index]), self.labels[index]
