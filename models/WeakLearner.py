import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torchmetrics import Accuracy

from training.ExperimentParameters import ExperimentParameters


class WeakLearner(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        self.params: ExperimentParameters = ExperimentParameters.from_dict(config)
        # Model, Loss, Optimizer
        self.model = self.params.get_model()
        self.model.to(self.device)

        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)

        self.val_loss = []
        self.val_accuracy = []

        self.test_loss = []
        self.test_accuracy = []

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optim = self.params.get_optimizer(self.model.parameters())

        scheduler = self.params.get_scheduler(optim)

        if scheduler is None:
            return optim
        else:
            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }

            match self.params.scheduler_parameters.scheduler_name:
                case "CosineAnnealingLR":
                    pass
                case "None":
                    pass
                case _:
                    raise NotImplementedError("Unsupported scheduler.")

            return {"optimizer": optim,
                    "lr_scheduler": lr_scheduler_config}

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

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

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.val_loss.append(loss)
        self.val_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        avg_acc = torch.stack(self.val_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.val_loss.clear()
        self.val_accuracy.clear()

    def on_fit_end(self):
        destroy_process_group()
