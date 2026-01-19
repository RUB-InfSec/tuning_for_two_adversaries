import itertools
import logging
import os
import sys
from typing import List, Tuple

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import yaml
from autoattack import AutoAttack, checks
from ray.tune import ExperimentAnalysis, ResultGrid
from rich.progress import track

from evaluation.AdversarialDataModule import AdversarialDataModule
from evaluation.EvaluatedExperiment import EvaluatedExperiment
from models.AverageLogitVoteEnsembleModel import AverageLogitVoteEnsembleModel
from models.WeakLearner import WeakLearner
from training.ExperimentParameters import ExperimentParameters
from training.TrainEngine import CustomDataModule
from utils.utils import AttackType
from utils.generate_config import cifar10_query_budgets
from utils.utils import find_key_paths, get_nested_attr, to_matrix

logger = logging.getLogger(__name__)

metric_column_names = ('model', 'metrics', 'CA', 'RA_transfer', 'RA_query')

def evaluate(args, ensemble_grouping_keys):
    eval_data_path = os.path.join("data", "processed_data", args.experiment_name)

    load_existing_data = False
    load_existing_query_data = False

    if not os.path.exists(eval_data_path):
        os.makedirs(eval_data_path, exist_ok=True)

    for k, v in args.opt_args['transfer_file'].items():
        assert '.pth' in v

    options_string = f"{'ensembles'}_{str(ensemble_grouping_keys)}"
    processed_data_path = os.path.join(eval_data_path,
                                       f"eval_object_{options_string}_{args.opt_args['config_file'].replace('.yaml', '')}.pth")
    query_file_path = os.path.join(eval_data_path, f"query_adversarial_samples_{options_string}.pth")

    if os.listdir(eval_data_path):
        if os.path.isfile(processed_data_path):
            if not args.force:
                logger.info(
                    "Evaluation data found. Re-evaluating based on this data. Please use --force if you want to override the evaluation data.")
                load_existing_data = True
            else:
                logger.warning("Experiment data found. Forcing an override of the underlying evaluation data...")
        if os.path.isfile(query_file_path):
            if not args.force:
                logger.info(
                    "Query data found. Continuing evaluation based on this data.")
                load_existing_query_data = True
            else:
                logger.warning("Experiment data found. Forcing an override of the underlying evaluation data...")
    else:
        # only empty dir found hence normal evaluation can commence.
        pass

    # disable message about current CUDA device to not interrupt tqdm
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

    # query budgets for square attack
    query_budgets = None

    if not load_existing_data:
        # identify the ablated parameter for plotting
        pathss = None
        if args.randomize:
            pathss = os.path.join("experiments", f"{'_'.join(args.experiment_name.split('_')[:5])}.yaml")
        else:
            pathss = os.path.join("experiments", f"{args.experiment_name}.yaml")

        with open(pathss, 'r') as ff:
            exp_params = yaml.safe_load(ff)

        df_weak, last_model_params, ablated_params, column_names = load_models(args, exp_params)

        # add some prefiltering, i.e., some forced fixed groups
        prefilters = []
        group_by = []

        for el in ensemble_grouping_keys:
            if el.startswith("-"):
                prefilters.append(el.replace("-", ""))
            else:
                group_by.append(el)

        # clean up the grid search parameters by the prefilters
        filtered_params = []
        filtered_params_idx = []
        for kdx, el in enumerate(ablated_params):
            if el in prefilters:
                filtered_params_idx.append(kdx)
            else:
                filtered_params.append(el)

        cd = CustomDataModule(last_model_params)
        cd.setup()

        if last_model_params['data_parameters']['dataset'] == 'cifar10':
            last_model_params['data_parameters']['batch_size'] = 128
            query_budgets = cifar10_query_budgets()
        else:
            raise NotImplementedError

        ensemble_data = []

        if len(group_by) == 0:
            logger.info("No evaluation keys found. Only evaluating weak learner.")
            df = df_weak
        else:
            # create grouped ensembles based on parameters, i.e., group all models (seeds, ...) for a specific combination of parameters, i.e., ensure homogeneity
            for param in group_by:
                if param == 'filtered_parameter_wise':
                    ablated_params = filtered_params
                    unique_params = [df_weak[par].unique() for par in ablated_params]

                    for combination in itertools.product(*unique_params):
                        mask = None
                        idf = [pd.NA] * len(ablated_params)
                        for jdx, com in enumerate(combination):
                            idf[jdx] = com
                            if mask is None:
                                mask = (df_weak[ablated_params[jdx]] == com)
                            else:
                                mask &= (df_weak[ablated_params[jdx]] == com)

                        # adjust idf
                        for idx in filtered_params_idx:
                            idf.insert(idx, pd.NA)

                        filtered_df = df_weak.loc[mask]
                        models = filtered_df['model'].to_list()

                        ens = AverageLogitVoteEnsembleModel(models)
                        ens.setup()

                        empty_cols = [pd.NA] * (len(metric_column_names) - 1)
                        ensemble_data += [tuple([ens.__class__.__name__] + idf + [ens] + empty_cols)]
                    break
                else:
                    # build ensembles over specific parameters (with diversification inside)
                    assert param in ablated_params

                    for val in df_weak[param].unique():
                        idf = [pd.NA] * len(ablated_params)
                        idf[ablated_params.index(param)] = val

                        filtered_df = df_weak[(df_weak[param] == val)]
                        models = filtered_df['model'].to_list()

                        ens = AverageLogitVoteEnsembleModel(models)
                        ens.setup()

                        empty_cols = [pd.NA] * (len(metric_column_names) - 1)
                        ensemble_data += [tuple([ens.__class__.__name__] + idf + [ens] + empty_cols)]

            df_ens = pd.DataFrame(ensemble_data, columns=column_names).sort_values(by=["model_type"] + ablated_params)
            df = pd.concat([df_weak, df_ens], axis=0, ignore_index=True)
            df = df[(df["model_type"] != "WeakLearner") & (df["model_type"].str.endswith("EnsembleModel"))]

    pl.seed_everything(42, workers=True)

    if load_existing_data:
        eval_object: EvaluatedExperiment = torch.load(processed_data_path, weights_only=False)
    else:
        for attack_type in [AttackType.QUERY, AttackType.TRANSFER]:
            ads: List[Tuple[int, AdversarialDataModule]] = []

            if attack_type == AttackType.QUERY:
                if load_existing_query_data:
                    ads = torch.load(query_file_path, weights_only=False)
                else:
                    small_test_dataloader = cd.small_test_dataloader()
                    ads = generate_query_images(params=exp_params, df=df, dataloader=small_test_dataloader,
                                                query_budgets=query_budgets)
                    torch.save(ads, query_file_path)

            elif attack_type == AttackType.TRANSFER and not load_existing_data:
                # load all transfer samples
                for k, v in args.opt_args['transfer_file'].items():
                    ad: AdversarialDataModule = AdversarialDataModule(meta_data=k, file_name=v)
                    ad.setup()
                    ads.append((0, ad))

            test_dataloader = cd.test_dataloader()
            if attack_type == AttackType.TRANSFER:
                adv_modules = ads
            elif attack_type == AttackType.QUERY:
                adv_modules = to_matrix(ads, len(query_budgets))
            else:
                raise NotImplementedError

            df = compute_robustness(attack_typ=attack_type, df=df, test_dataloader=test_dataloader,
                                    adv_modules=adv_modules)

        # remove the models, as we do not want to store them all in the dataframe
        df = df.drop(['model'], axis=1)
        eval_object: EvaluatedExperiment = EvaluatedExperiment(name=args.experiment_name, config=exp_params, data=df)
        torch.save(eval_object, processed_data_path)

    return eval_object


def load_models(args, exp_params):
    paths = find_key_paths(exp_params, key_value="grid_search")

    result = ResultGrid(ExperimentAnalysis(
        experiment_checkpoint_path=os.path.join(os.getcwd(), "data/trained_models", args.experiment_name)))

    column_names = ["model_type"]

    data = []
    last_finished_trial = None
    errored = 0

    for res in result:
        if res.error is not None:
            errored += 1
            continue

        last_finished_trial = res
        best_params = ExperimentParameters.from_dict(res.config['train_loop_config'])
        metrics_dataframe = res.metrics_dataframe

        best_checkpoints = [
            res.get_best_checkpoint(metric="ptl/val_loss", mode='min')]

        for bc in best_checkpoints:
            with bc.as_directory() as checkpoint_dir:
                file = os.path.join(checkpoint_dir, 'checkpoint.ckpt')
                if os.path.getsize(file) > 0:
                    model = WeakLearner.load_from_checkpoint(file)
                    # disable randomness, dropout, etc...
                    model.eval()

                    mod_params, values = get_nested_attr(model.params, paths)

                    if len(column_names) == 1:
                        ablated_params = mod_params
                        column_names += mod_params
                        column_names += metric_column_names

                    data += [tuple([model.__class__.__name__] + values + [model, metrics_dataframe.iloc[
                        int(bc.path.split('/')[-1].split('_')[-1])]] + [pd.NA] * (len(metric_column_names) - 2))]

    if errored > 0:
        logger.error(f"---- Encountered {errored} errored trials. ----")
        sys.exit(0)

    df = pd.DataFrame(data, columns=column_names).sort_values(by=["model_type"] + ablated_params)

    return df, last_finished_trial.config['train_loop_config'], ablated_params, column_names


def compute_robustness(attack_typ, df, test_dataloader, adv_modules):
    trainer = pl.Trainer(
        deterministic=True,
        benchmark=False,
        devices=[0],
        accelerator="gpu",
        enable_progress_bar=False,
        num_nodes=1,
        enable_model_summary=False
    )

    for idx, (row, mod) in enumerate(
            track(list(df['model'].items()), description=f"Evaluating {attack_typ.value} robustness...")):
        mod.cuda()

        if attack_typ == AttackType.QUERY:
            adv_module = adv_modules[idx]
        elif attack_typ == AttackType.TRANSFER:
            adv_module = adv_modules

        ras = []

        for kdx, module in adv_module:
            if attack_typ == AttackType.QUERY:
                # make sure that the computed query images still belong to the correct model
                assert row == kdx

            adv_loader = module.small_test_dataloader(size=1000)

            ra = trainer.test(mod, dataloaders=adv_loader, verbose=False)[0]
            ras.append(ra['ptl/test_accuracy'])

        df.at[row, f'RA_{attack_typ.value}'] = ras

        ca_val = df.at[row, 'CA']
        is_missing = pd.isna(ca_val) if not isinstance(ca_val, (list, np.ndarray)) else False

        if is_missing and test_dataloader is not None:
            if not isinstance(test_dataloader, list):
                test_dataloader = [test_dataloader]

            cas = []
            for loader in test_dataloader:
                ca = trainer.test(mod, dataloaders=loader, verbose=False)[0]
                cas.append(ca['ptl/test_accuracy'])

            df.at[row, 'CA'] = cas

    return df


def generate_query_images(params, df, dataloader, query_budgets, normalize=True):
    epsilon = 8/255

    if normalize:
        config = ExperimentParameters.from_dict(params)
        normalize = config.data_parameters.test_transforms.transforms[-1]
        std = torch.tensor(normalize.std).cuda()
        mean = torch.tensor(normalize.mean).cuda()

        denormalize = transforms.Lambda(lambda x: x * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1))

    # enable debugging
    def blocked_function(*args, **kwargs):
        return None

    # replace the function with the blocked version to enable debugging
    checks.check_dynamic = blocked_function
    # disable complaints about ensembles
    checks.check_range_output = blocked_function

    ads: List[Tuple[int, AdversarialDataModule]] = []

    for idx, (row, mod) in enumerate(
            track(list(df['model'].items()), description="Generating query-based adversarial examples...")):
        for budget in query_budgets:
            # takes [0,1] in NCHW format and return logits
            def forward_pass(images):
                if normalize:
                    return mod(normalize(images))
                else:
                    return mod(images)

            adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard', verbose=False)
            adversary.seed = 42
            adversary.attacks_to_run = ['square']
            adversary.square.n_queries = budget

            images, labels = next(iter(dataloader))

            if normalize:
                images_denorm = denormalize(images.to(mod.device))
            else:
                images_denorm = images.to(mod.device)

            labels = labels.to(mod.device)

            adv_img = adversary.run_standard_evaluation_individual(images_denorm, labels, bs=len(images))

            # Prepare for adversarial dataloader
            ad: AdversarialDataModule = AdversarialDataModule(attack_type="square",
                                                              meta_data=str(budget),
                                                              adversarial_images=adv_img['square'],
                                                              labels=labels)
            ad.setup()
            ads.append((row, ad))

    return ads