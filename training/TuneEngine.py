import copy
import logging
import os
from typing import List, Tuple

import optuna
import pandas as pd
import ray
import ray.tune as tune
from ray.air import RunConfig
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch

from evaluation.AdversarialDataModule import AdversarialDataModule
from evaluation.evaluate import generate_query_images, compute_robustness, metric_column_names
from models.AverageLogitVoteEnsembleModel import AverageLogitVoteEnsembleModel
from training.CustomDataModule import CustomDataModule
from training.TrainEngine import train
from models.WeakLearner import WeakLearner
from utils.utils import AttackType
from utils.utils import to_matrix

logger = logging.getLogger(__name__)


def multi_objective(config, args, params):
    params["optimizer_parameters"]["learning_rate"] = config["search_space"]["learning_rate"]
    params["optimizer_parameters"]["momentum"] = config["search_space"]["momentum"]
    params["optimizer_parameters"]["weight_decay"] = config["search_space"]["weight_decay"]
    params["data_parameters"]["batch_size"] = config["search_space"]["batch_size"]

    results = train(args, params)

    column_names = ["model_type"]
    data = []
    last_finished_trial = None

    models = []

    for res in results:
        bc = res.get_best_checkpoint(metric="ptl/val_loss", mode="min")
        metrics_dataframe = res.metrics_dataframe

        last_finished_trial = res

        with bc.as_directory() as checkpoint_dir:
            file = os.path.join(checkpoint_dir, 'checkpoint.ckpt')
            if os.path.getsize(file) > 0:
                model = WeakLearner.load_from_checkpoint(file)
                # disable randomness, dropout, etc...
                model.eval()

                data += [tuple([model.__class__.__name__, model, metrics_dataframe.iloc[
                    int(bc.path.split('/')[-1].split('_')[-1])]] + [pd.NA] * (len(metric_column_names) - 2))]

                models.append(model)

        df = pd.DataFrame(data, columns=column_names + list(metric_column_names)).sort_values(by=["model_type"])

    if len(models) > 1:
        # wrap as ensemble for multiple WeakLearners
        model = AverageLogitVoteEnsembleModel(df['model'].to_list())
        model.setup()

        data = [tuple([model.__class__.__name__, model] + [pd.NA] * (len(metric_column_names) - 1))]
        df = pd.DataFrame(data, columns=column_names + list(metric_column_names)).sort_values(by=["model_type"])

    last_model_params = last_finished_trial.config['train_loop_config']

    cd = CustomDataModule(last_model_params)
    cd.setup()

    if last_model_params['data_parameters']['dataset'] == 'cifar10':
        last_model_params['data_parameters']['batch_size'] = 128
        query_budgets = [500]

    for attack_type in [AttackType.QUERY, AttackType.TRANSFER]:
        ads: List[Tuple[int, AdversarialDataModule]] = []

        if attack_type == AttackType.QUERY:
            small_test_dataloader = cd.small_test_dataloader(size=1000)
            ads = generate_query_images(params=params, df=df, dataloader=small_test_dataloader,
                                        query_budgets=query_budgets)

        elif attack_type == AttackType.TRANSFER:
            # load all transfer samples
            for k, v in args.opt_args['transfer_file'].items():
                ad: AdversarialDataModule = AdversarialDataModule(meta_data=k, file_name=v)
                ad.setup()
                ads.append((0, ad))

        if attack_type == AttackType.TRANSFER:
            adv_modules = ads
        elif attack_type == AttackType.QUERY:
            adv_modules = to_matrix(ads, len(query_budgets))
        else:
            raise NotImplementedError

        df = compute_robustness(attack_typ=attack_type, df=df,
                                test_dataloader=[cd.val_dataloader(), cd.test_dataloader()], adv_modules=adv_modules)

    validation_accuracy = df.loc[0, 'CA'][0]
    test_accuracy = df.loc[0, 'CA'][1]

    ra_transfer = df.loc[0, 'RA_transfer'][-1]
    ra_query = df.loc[0, 'RA_query'][0]
    robust_accuracy = min(ra_transfer, ra_query)

    ray.train.report({
        "optuna/val_accuracy": validation_accuracy, "optuna/robust_accuracy": robust_accuracy,
        "optuna/test_accuracy": test_accuracy,
        "optuna/ra_transfer": ra_transfer, "optuna/ra_query": ra_query
    })


def tune_it(args, params):
    searcher = OptunaSearch(sampler=optuna.samplers.NSGAIISampler(population_size=20, mutation_prob=None,
                                                                  crossover=None,
                                                                  crossover_prob=0.9, swapping_prob=0.5,
                                                                  seed=42),
                            metric=["optuna/val_accuracy", "optuna/robust_accuracy"], mode=["max", "max"])

    search_space = {
        "learning_rate": tune.loguniform(1e-4, 4e-1),
        "momentum": tune.uniform(0.8, 1.0),
        "weight_decay": tune.loguniform(1e-06, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048])
    }

    reporter = CLIReporter(
        metric_columns=["optuna/val_accuracy", "optuna/robust_accuracy"],
        parameter_columns=["search_space/learning_rate", "search_space/momentum", "search_space/weight_decay",
                           "search_space/batch_size"],
        max_report_frequency=120
    )

    def make_objective(args, base_params):
        def wrapped(config):
            return multi_objective(
                config=config,
                args=args,
                params=base_params
            )

        return wrapped

    tuner = tune.Tuner(
        trainable=tune.with_resources(
            make_objective(args=args, base_params=copy.deepcopy(params)),
            resources={"cpu": 4, "gpu": 0.025}
        ),
        tune_config=tune.TuneConfig(
            search_alg=searcher,
            num_samples=100,
            max_concurrent_trials=1
        ),
        run_config=RunConfig(
            name=f"optuna_tuning_{args.experiment_name.split('/')[-1]}",
            storage_path=os.path.join(os.getcwd(), "data", 'tuned_models'),
            progress_reporter=reporter),
        param_space={"search_space": search_space},
    )

    results = tuner.fit()

    best_val = results.get_best_result("optuna/val_accuracy", "max")
    best_racc = results.get_best_result("optuna/robust_accuracy", "max")

    print(
        f"Best hyperparameters for val_accuracy found were: {best_val.config} @ CA={best_val.metrics['optuna/val_accuracy']:.02f}, RA={best_val.metrics['optuna/robust_accuracy']:.02f}")
    print()
    print(
        f"Best hyperparameters for robust_accuracy found were: {best_racc.config} @ CA={best_racc.metrics['optuna/val_accuracy']:.02f}, RA={best_racc.metrics['optuna/robust_accuracy']:.02f}")
