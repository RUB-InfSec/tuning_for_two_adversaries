import logging
import os
from argparse import ArgumentParser

import lightning.pytorch as pl
import ray.tune as tune
import torch
from git import Repo, InvalidGitRepositoryError
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.lightning import (
    RayLightningEnvironment,
    prepare_trainer, RayDDPStrategy, RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer

from training.CustomDataModule import CustomDataModule
from training.IntervalRayTrainReportCallback import IntervalRayTrainReportCallback
from models.WeakLearner import WeakLearner

logger = logging.getLogger(__name__)


def train_func(config):
    if 'randomize' in config:
        config['network_parameters']['initialization_seed'] += (config['randomize'] + 1) * 100

    pl.seed_everything(config['network_parameters']['initialization_seed'], workers=False)
    model = WeakLearner(config)

    pl.seed_everything(config['data_parameters']['batching_seed'], workers=True)
    dm = CustomDataModule(config)

    pl.seed_everything(config['global_seed'], workers=False)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Ray sets this
    gpu_id = local_rank % torch.cuda.device_count()  # Ensure correct mapping

    # we always checkpoint such that the minimum validation checkpoint can also be saved
    checkpointing_interval = -1

    callbacks = [LearningRateMonitor(logging_interval='step', log_momentum=True, log_weight_decay=True)]

    if config['tuner']:
        callbacks.append(RayTrainReportCallback())
        callbacks.append(EarlyStopping(monitor="ptl/val_loss", min_delta=0.00, patience=10, verbose=False, mode="min"))
    else:
        callbacks.append(IntervalRayTrainReportCallback(checkpointing_interval=checkpointing_interval))

    trainer = pl.Trainer(
        deterministic=True,
        benchmark=False,
        devices=[gpu_id],
        accelerator="gpu",
        strategy=RayDDPStrategy(),
        callbacks=callbacks,
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs=config['training_parameters']['epochs'],
        num_nodes=1,
        enable_model_summary=False,
        # speed ups for experiments of higher complexity
        precision="bf16-mixed" if (config['tuner'] or config['data_parameters'][
            'node_count'] == 1) else '32-true'
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


def get_trainer(run_cfg: RunConfig, tune=False) -> TorchTrainer:
    if tune:
        scaling_config = ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker={"GPU": 0.05, "CPU": 8}
        )
    else:
        scaling_config = ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 24}
        )
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_cfg,
    )
    return ray_trainer


def train(args: ArgumentParser, params: dict):
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="ptl/val_loss",
        checkpoint_score_order="min",
    )

    if args.mode == "run_rq3":
        storage_path = os.path.join(os.getcwd(), "data", "tuned_models")
    else:
        try:
            repo = Repo(os.getcwd())
            # Get the latest commit on the current HEAD
            latest_commit = repo.head.commit
            params['train_git_hash'] = latest_commit.hexsha
        except InvalidGitRepositoryError:
            logger.log(logging.INFO, "No Git repository found. Omitting commit hash from log.")

        storage_path = os.path.join(os.getcwd(), "data", "trained_models")

    params.update({'tuner': args.mode == 'run_rq3'})

    # Due to the way the experiments are executed the deep ensembles require a dedicated initialization seed to prevent
    # overlaps with different global seeds. This problem does not exist for the other ML instances.
    if params['data_parameters']['node_count'] == 1:
        params.update({'randomize': args.seed})

    exp_name = args.experiment_name

    run_config = RunConfig(
        storage_path=storage_path, verbose=0,
        name=exp_name,
        log_to_file=True,
        checkpoint_config=checkpoint_config
    )

    tuner = tune.Tuner(
        get_trainer(run_cfg=run_config, tune=params['tuner']),
        param_space={"train_loop_config": params},
    )

    result = tuner.fit()

    return result
