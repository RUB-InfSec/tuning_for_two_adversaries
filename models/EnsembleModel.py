import copy
from abc import ABC, abstractmethod
from typing import List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

from models.WeakLearner import WeakLearner


class EnsembleModel(pl.LightningModule, ABC):
    models: List[WeakLearner] = []

    def __init__(self, models: List[WeakLearner]):
        super().__init__()

        self.models = copy.deepcopy(models)
        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)

        self.test_loss = []
        self.test_accuracy = []

    def setup(self, stage=None):
        self.to(self.models[0].device)

    @abstractmethod
    def forward(self, x):
        pass

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.test_loss.append(loss)
        self.test_accuracy.append(accuracy)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_loss).mean()
        avg_acc = torch.stack(self.test_accuracy).mean()
        self.log("ptl/test_loss", avg_loss, sync_dist=True)
        self.log("ptl/test_accuracy", avg_acc, sync_dist=True)
        self.test_loss.clear()
        self.test_accuracy.clear()
