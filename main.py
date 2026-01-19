import argparse
import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict

import lightning.pytorch as pl
import seaborn as sns

from evaluation.plotter import plot_rq12, plot_rq3, plot_rq12_individual

sns.set_palette("colorblind")
import ray
import ray.tune as tune
import torch
import yaml
from ray.tune.search.sample import Integer

from evaluation.evaluate import evaluate
from training import TrainEngine, TuneEngine

from utils.generate_config import generate_experiment_config

import json

logger = logging.getLogger(__name__)

def setup(args: ArgumentParser):
    os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"] = "0"
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "32"

    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    os.makedirs("data/attacks", exist_ok=True)
    os.makedirs("data/processed_data", exist_ok=True)
    os.makedirs("data/trained_models", exist_ok=True)
    os.makedirs("data/plots", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    torch.set_float32_matmul_precision(precision="high")

    if args.mode in ["train", "tune"]:
        ray.init(
            configure_logging=True,
            logging_level=logging.INFO,
            include_dashboard=False,
            _temp_dir=os.path.join(os.getcwd(), 'ray_tmp')
        )

    pl.seed_everything(args.seed, workers=True, verbose=False)

    def tune_randint_representer(dumper, data):
        return dumper.represent_mapping('!tune.randint', {'lower': data.lower, 'upper': data.upper})

    def tune_randint_constructor(loader, node):
        node_dict = {}
        for key, value in node.value:
            node_dict.update({loader.construct_object(key): loader.construct_object(value, deep=True)})
        return tune.randint(lower=node_dict['lower'], upper=node_dict['upper'])

    yaml.SafeDumper.add_representer(Integer, tune_randint_representer)
    yaml.SafeLoader.add_constructor('!tune.randint', tune_randint_constructor)

    def ordered_dict_representer(self, value):  # can be a lambda if that's what you prefer
        return self.represent_mapping('tag:yaml.org,2002:map', value.items())

    yaml.SafeDumper.add_representer(OrderedDict, ordered_dict_representer)


def parse_args() -> ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default=None,
                        choices=['create_config_rq12', 'run_rq12', 'run_rq3', 'evaluate_rq12', 'plot_rq12', 'plot_rq12_individual',
                                 'plot_rq3'],
                        help="Mode to run.")
    parser.add_argument("--experiment_name", default=None, help="Experimental configuration to use for the experiment.")
    parser.add_argument("--gpus", default="0", help="Comma separated list of GPUs to use. Sorted by PCI_BUS_ID.")
    parser.add_argument("--seed", default="42", type=int, help="Global seed for seeding all other seeds.")
    parser.add_argument("--opt_args", nargs='*', type=str, help="Optional arguments.")

    parser.add_argument("--randomize", action="store_true",
                        help="Triggers three randomized repetitions of an experiment.")
    parser.add_argument("--force", action="store_true", help="Overrides already evaluated experiments.")

    args = parser.parse_args()

    return args


def validate_transfer_file(config_file):
    attacks = 'MI_CommonWeakness'
    config_path = os.path.join(os.getcwd(), "data", "attacks", config_file)

    file_paths = {}
    surrs = None

    assert os.path.exists(config_path)

    with open(config_path, 'r') as ff:
        transfer_config = yaml.safe_load(ff)

    surrs = transfer_config['surrogates']

    file_path = os.path.join(os.path.dirname(config_path), transfer_config['file'])
    assert os.path.exists(file_path)

    file_paths.update({attacks: file_path})

    return file_paths, transfer_config['surrogates']


def load_opt_args(arg):
    if args.opt_args is None:
        with open(os.path.join("experiments", f"{args.experiment_name}.yaml"), 'r') as ff:
            exp_params = yaml.safe_load(ff)

        args.opt_args = exp_params['evaluation_parameters']
    else:
        try:
            if isinstance(args.opt_args, list):
                args.opt_args = json.loads(args.opt_args[0])
            else:
                args.opt_args = json.loads(args.opt_args)
            print(args.opt_args)
        except json.JSONDecodeError:
            logger.critical("Invalid JSON format. Please provide a valid JSON string.")
            return

    if arg.opt_args['transfer_file'] != '':
        local_file_paths, surrogates = validate_transfer_file(config_file=args.opt_args['transfer_file'])

        args.opt_args.update({'config_file': args.opt_args['transfer_file']})
        args.opt_args.update({'transfer_file': local_file_paths})
        args.opt_args.update({'surrogate_files': surrogates})


def sanity_checks(args: ArgumentParser):
    if args.experiment_name is not None:
        if args.mode != 'create_config_rq12':
            args.experiment_name = args.experiment_name[args.experiment_name.index('experiments') + len('experiments') + 1:]
            args.experiment_name = args.experiment_name.replace(".yaml", "")
            assert os.path.exists(os.path.join(os.getcwd(), 'experiments', f"{args.experiment_name}.yaml"))

    if args.mode == 'run_rq12' or args.mode == 'run_rq3':
        assert args.experiment_name is not None
        load_opt_args(args)
    elif args.mode == "evaluate_rq12":
        # make sure evaluation is just done on one GPU to avoid concurrency problems
        assert len(args.gpus.split(",")) == 1

        load_opt_args(args)
        # assert args.opt_args['ensemble_keys']

        if not args.randomize:
            assert os.path.exists(os.path.join(os.getcwd(), "data", "trained_models", args.experiment_name))
    elif args.mode == "plot_rq12_individual":
        assert args.experiment_name is not None

if __name__ == "__main__":
    args = parse_args()
    setup(args)
    sanity_checks(args)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                            logging.FileHandler("output.log"),
                            logging.StreamHandler()
                        ])

    if args.mode == 'create_config_rq12':
        config_path = os.path.join("experiments", f"{args.experiment_name}.yaml")
        if not os.path.exists(config_path):
            pass
        elif not args.force:
            logger.info(
                "Config already exists. Please use --force if you want to override this config.")
            exit(0)
        else:
            logger.warning("Config already exists. Overwriting...")

        params_dict = generate_experiment_config()
        with open(config_path, 'w+') as ff:
            yaml.safe_dump(OrderedDict(params_dict), ff)

    elif args.mode == 'run_rq12':
        with open(os.path.join("experiments", f"{args.experiment_name}.yaml"), 'r') as ff:
            exp_params = yaml.safe_load(ff)

        if args.randomize:
            original_name = args.experiment_name
            for seed in [0, 1, 42]:
                logger.info(f"Setting seed {seed}...")
                pl.seed_everything(seed, workers=True, verbose=False)
                args.seed = seed
                args.experiment_name = f"{original_name}_{seed}"

                if os.path.exists(os.path.join(os.getcwd(), "data", "trained_models", args.experiment_name)):
                    logger.info("Run already exists. Continuing...")
                else:
                    TrainEngine.train(args, exp_params)
        else:
            TrainEngine.train(args, exp_params)

    elif args.mode == "run_rq3":
        with open(os.path.join("experiments", f"{args.experiment_name}.yaml"), 'r') as ff:
            exp_params = yaml.safe_load(ff)

        TuneEngine.tune_it(args, exp_params)
    elif args.mode == "evaluate_rq12":
        if args.randomize:
            objs = []
            for seed in [0, 1, 2]:
                logger.info(f"Setting seed {seed}...")
                pl.seed_everything(seed, workers=True, verbose=False)
                args.seed = seed
                args.experiment_name = f"{args.experiment_name}_{seed}"

                eval_obj = evaluate(args, ensemble_grouping_keys=args.opt_args['ensemble_keys'])
                objs.append(eval_obj)
        else:
            eval_obj = evaluate(args, ensemble_grouping_keys=args.opt_args['ensemble_keys'])

    elif args.mode == "plot_rq12":
        plot_rq12()
    elif args.mode == "plot_rq12_individual":
        plot_rq12_individual(args)
    elif args.mode == "plot_rq3":
        plot_rq3()
