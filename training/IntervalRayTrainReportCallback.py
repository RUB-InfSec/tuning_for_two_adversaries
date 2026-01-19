import os
import shutil
import tempfile
from pathlib import Path

import lightning.pytorch as pl
from ray import train
from ray.train import Checkpoint


class IntervalRayTrainReportCallback(pl.callbacks.Callback):

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(self, checkpointing_interval: int) -> None:
        super().__init__()

        self.checkpointing_interval = checkpointing_interval

        self.trial_name = train.get_context().get_trial_name()
        self.local_rank = train.get_context().get_local_rank()
        self.tmpdir_prefix = Path(tempfile.gettempdir(), self.trial_name).as_posix()
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.checkpointing_interval > 0:
            should_checkpoint = trainer.current_epoch != 0 and (
                    (trainer.current_epoch % self.checkpointing_interval) == 0 or (
                    trainer.current_epoch == trainer.max_epochs - 1))
        else:
            should_checkpoint = trainer.current_epoch >= 0

        # Creates a checkpoint dir with fixed name
        tmpdir = Path(self.tmpdir_prefix, str(trainer.current_epoch)).as_posix()
        os.makedirs(tmpdir, exist_ok=True)

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        checkpoint = None

        if should_checkpoint:
            # Save checkpoint to local
            ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
            trainer.save_checkpoint(ckpt_path, weights_only=False)

            # Report to train session
            checkpoint = Checkpoint.from_directory(tmpdir)

        train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        trainer.strategy.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)
