import copy
import os

import matplotlib.pyplot as plt
import pyarrow
import seaborn as sns
from matplotlib.ticker import MultipleLocator

sns.set_palette("colorblind")
import torch
from ray.tune import ResultGrid, ExperimentAnalysis
import pandas as pd
from evaluation.EvaluatedExperiment import EvaluatedExperiment
from utils.utils import get_pareto_front
from utils.utils import AttackType

def plot_rq12_individual(args):
    eval_data_path = os.path.join("data", "processed_data", args.experiment_name)

    assert os.path.exists(eval_data_path)

    files = []
    for file in os.listdir(eval_data_path):
        if file.startswith('eval_object'):
            files.append(file)

    assert len(files) == 1

    eval_object: EvaluatedExperiment = torch.load(os.path.join(eval_data_path, files[0]),
                                                  weights_only=False)

    data = copy.deepcopy(eval_object.data)
    data['CA'] = data['CA'].apply(
        lambda x: x[-1] if isinstance(x, list) else None
    )
    data['RA_query'] = data['RA_query'].apply(
        lambda x: x[-1] if isinstance(x, list) else None
    )
    data['RA_transfer'] = data['RA_transfer'].apply(
        lambda x: x[-1] if isinstance(x, list) else None
    )

    data = data[(data['model_type'] == 'AverageLogitVoteEnsembleModel')]
    data = data.drop(['model_type', 'metrics'], axis=1)

    key = data.columns.to_list()[0]
    names = {'optimizer_parameters.learning_rate' : r'$\eta$', 'data_parameters.batch_size' : r'$B$', 'optimizer_parameters.momentum' : r'$\mu$',
            'optimizer_parameters.weight_decay' : r'$\lambda$'}

    for attack in [AttackType.TRANSFER, AttackType.QUERY]:
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Plot RA_transfer on left y-axis
        sns.lineplot(
            data=data,
            x=key,
            y=f'RA_{attack.value}',
            ax=ax1,
            marker=None,
            color='blue',
            linewidth=3,
            markersize=8,
        )

        ax1.grid(True, which="both", ls="-")
        ax1.set_ylabel(r'$\mathsf{RA}_\mathrm{' + (r'Q' if attack.value == "query" else r'T') + r'}$')
        ax1.set_xlabel(
            f'{names[key]} {"(log scale)" if key != "optimizer_parameters.momentum" and key != "data_parameters.batch_size" else ""}')

        if key != 'optimizer_parameters.momentum':
            if key == 'data_parameters.batch_size':
                ax1.set_xscale('log', base=2)
            else:
                ax1.set_xscale('log')

        # Plot CA on right y-axis
        ax2 = ax1.twinx()

        sns.lineplot(
            data=data,
            x=key,
            y='CA',
            ax=ax2,
            marker=None,
            linewidth=3,
            color='orange',
            markersize=8,
        )

        ax2.grid(True, which="both", ls="-")
        ax2.set_ylabel(r'$\mathsf{CA}$')

        plt.show()


def plot_rq12():
    ablations = ['lr', 'bs', 'mom', 'wd']

    ablations_keys = {'wd': 'optimizer_parameters.weight_decay', 'lr': 'optimizer_parameters.learning_rate',
                      'bs': 'data_parameters.batch_size', 'mom': 'optimizer_parameters.momentum'}

    objs = [[] for _ in range(len(ablations))]

    for i, abl in enumerate(ablations):
        outer_path = os.path.join(os.getcwd(), 'data', 'processed_data', 'cifar10', abl)
        op = sorted(os.listdir(outer_path))

        # three iterations for 10 different experiments
        assert len(op) == 30

        for tx in range(0, len(op), 3):
            exps = tuple(op[tx:tx + 3])
            placeholder = None
            averaged_df = []

            for exi in exps:
                inner_path = os.path.join(outer_path, exi)
                for file in os.listdir(inner_path):
                    if file.startswith('eval_object'):
                        eval_object: EvaluatedExperiment = torch.load(os.path.join(inner_path, file),
                                                                      weights_only=False)
                        if placeholder is None:
                            placeholder = eval_object
                            placeholder.name = placeholder.name.replace('_0', '')

                        data = copy.deepcopy(eval_object.data)
                        data['CA'] = data['CA'].apply(
                            lambda x: x[-1] if isinstance(x, list) else None
                        )
                        data['RA_query'] = data['RA_query'].apply(
                            lambda x: x[-1] if isinstance(x, list) else None
                        )
                        data['RA_transfer'] = data['RA_transfer'].apply(
                            lambda x: x[-1] if isinstance(x, list) else None
                        )

                        data = data[(data['model_type'] == 'AverageLogitVoteEnsembleModel')]
                        data = data[[ablations_keys[abl], 'CA', 'RA_transfer', 'RA_query']]
                        averaged_df.append(data)

            placeholder.data = (sum(averaged_df) / len(averaged_df)).round(7)
            objs[i].append(copy.deepcopy(placeholder))

    # pre-iterate and average results
    dataframes = [[] for _ in range(len(ablations))]
    for jdx, abl in enumerate(objs):
        for kdx, exp in enumerate(abl):
            namer = exp.name
            ids = exp.name.split('/')[-1].split('_')[-3:]
            data = exp.data
            data['node_count'] = int(ids[0])
            data['split_type'] = 'full' if ids[1] == "full" else "dist"

            dataframes[ablations.index(ids[2])].append(data)

    for jdx, abl in enumerate(dataframes):
        dataframes[jdx] = pd.concat(dataframes[jdx])

    keys = ['optimizer_parameters.learning_rate', 'data_parameters.batch_size', 'optimizer_parameters.momentum',
            'optimizer_parameters.weight_decay']
    names = [r'$\eta$', r'$B$', r'$\mu$', r'$\lambda$']

    fontsize = 28

    for jdx, abl in enumerate(dataframes):
        for attack in [AttackType.TRANSFER, AttackType.QUERY]:

            plottt = dataframes[jdx][dataframes[jdx]['split_type'].isin(['full', 'dist'])]

            df_long = plottt.melt(
                id_vars=[keys[jdx], 'split_type'],
                value_vars=['CA', f'RA_{attack.value}'],
                var_name='metric',
                value_name='accuracy'
            )

            df_mean = df_long.groupby(
                [keys[jdx], 'split_type', 'metric']
            )['accuracy'].mean().reset_index()

            line_styles = {
                'full': '-',
                'dist': '--',
            }

            fig, ax1 = plt.subplots(figsize=(8, 5))

            # Plot RA_transfer on left y-axis
            for split_type, line in line_styles.items():
                data_ra = df_mean[
                    (df_mean['split_type'] == split_type) & (df_mean['metric'] == f'RA_{attack.value}')]
                sns.lineplot(
                    data=data_ra,
                    x=keys[jdx],
                    y='accuracy',
                    ax=ax1,
                    linestyle=line,
                    marker=None,
                    color='blue',
                    linewidth=3,
                    markersize=8,
                    label=f'RA ({split_type})',
                )

            ax1.grid(True, which="both", ls="-")
            ax1.set_ylabel(r'$\mathsf{RA}_\mathrm{' + (r'Q' if attack.value == "query" else r'T') + r'}$',
                           fontsize=fontsize)
            ax1.set_xlabel(
                f'{names[jdx]} {"(log scale)" if keys[jdx] != "optimizer_parameters.momentum" and keys[jdx] != "data_parameters.batch_size" else ""}',
                fontsize=fontsize)

            if keys[jdx] != 'optimizer_parameters.momentum':
                if keys[jdx] == 'data_parameters.batch_size':
                    ax1.set_xscale('log', base=2)
                else:
                    ax1.set_xscale('log')

            ax1.tick_params(axis='both', labelsize=fontsize - 2)

            # Plot CA on right y-axis
            ax2 = ax1.twinx()
            for split_type, line in line_styles.items():
                data_ca = df_mean[(df_mean['split_type'] == split_type) & (df_mean['metric'] == 'CA')]
                sns.lineplot(
                    data=data_ca,
                    x=keys[jdx],
                    y='accuracy',
                    ax=ax2,
                    marker=None,
                    linestyle=line,
                    linewidth=3,
                    color='orange',
                    markersize=8,
                    label=f'CA ({split_type})',
                )

            ax2.grid(True, which="both", ls="-")
            ax2.set_ylabel(r'$\mathsf{CA}$', fontsize=fontsize)

            low, high = ax2.get_ylim()
            ax2.set_ylim(low, 1.0)

            if jdx in [2, 3]:
                if attack.value == 'query':
                    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
                else:
                    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
                ax2.yaxis.set_major_locator(MultipleLocator(0.1))
            elif jdx == 1:
                if attack.value == 'query':
                    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
                else:
                    ax1.yaxis.set_major_locator(MultipleLocator(0.2))

            ax2.tick_params(axis='both', labelsize=fontsize - 2)

            ax1.get_legend().remove()
            ax2.get_legend().remove()

            fig.tight_layout()

            os.makedirs('data/plots/rq12', exist_ok=True)
            plt.savefig(f"data/plots/rq12/{attack.value}_{keys[jdx]}.pdf", format='pdf',
                        bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()


def plot_rq3():
    base_path = os.path.join(os.getcwd(), "data", 'tuned_models')
    fig, ax = plt.subplots(figsize=(11, 5))
    fontsize = 24

    labels = [r'Centralized ($N=1$)', r'ENS-Full ($N=3$)', r'ENS-Full ($N=5$)', r'ENS-Full ($N=7$)',
              r'ENS-IID ($N=3$)', r'ENS-IID ($N=5$)', r'ENS-IID ($N=7$)', r'ENS-NonIID ($N=3$)',
              r'ENS-NonIID ($N=5$)', r'ENS-NonIID ($N=7$)']

    exps = ['optuna_tuning_1_full', 'optuna_tuning_3_full', 'optuna_tuning_5_full', 'optuna_tuning_7_full',
            'optuna_tuning_3_uniform', 'optuna_tuning_5_uniform', 'optuna_tuning_7_uniform',
            'optuna_tuning_3_dirichlet', 'optuna_tuning_5_dirichlet', 'optuna_tuning_7_dirichlet']

    for idx, exp in enumerate(exps):
        if exp == 'tuning':
            continue
        try:
            results = ResultGrid(ExperimentAnalysis(
                experiment_checkpoint_path=os.path.join(base_path, f"{exp}")))
        except pyarrow.lib.ArrowInvalid:
            continue

        best_val = results.get_best_result("optuna/val_accuracy", "max")
        best_racc = results.get_best_result("optuna/robust_accuracy", "max")

        exp = exp.replace("optuna_tuning_", "")
        print(f"({exp})")

        print(
            f"Best params for val_accuracy: \t\t\t{best_val.config} @ CA={best_val.metrics['optuna/val_accuracy']:.02f}, RA_min={best_val.metrics['optuna/robust_accuracy']:.02f}, RA_t={best_val.metrics['optuna/ra_transfer']:.02f}, RA_q={best_val.metrics['optuna/ra_query']:.02f}")
        print(
            f"Best params for robust_accuracy: \t\t{best_racc.config} @ CA={best_racc.metrics['optuna/val_accuracy']:.02f}, RA_min={best_racc.metrics['optuna/robust_accuracy']:.02f}, RA_t={best_racc.metrics['optuna/ra_transfer']:.02f}, RA_q={best_racc.metrics['optuna/ra_query']:.02f}")

        objectives = ["optuna/val_accuracy", "optuna/robust_accuracy"]
        pareto_trials = get_pareto_front(results, objectives)

        filtered = False
        if filtered:
            pareto_trials = [trial for trial in pareto_trials if trial.metrics["optuna/val_accuracy"] >= 0.86]

        if len(pareto_trials) > 0 and filtered:
            # Find best trial for val_accuracy
            best_val_trial = max(pareto_trials, key=lambda t: t.metrics["optuna/val_accuracy"])

            # Find best trial for robust_accuracy
            best_racc_trial = max(pareto_trials, key=lambda t: t.metrics["optuna/robust_accuracy"])
            print(
                "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print(
                f"Best params (filtered) for val_accuracy: \t{best_val_trial.config} "
                f"@ CA={best_val_trial.metrics['optuna/val_accuracy']:.02f}, "
                f"RA_min={best_val_trial.metrics['optuna/robust_accuracy']:.02f}, "
                f"RA_t={best_val_trial.metrics['optuna/ra_transfer']:.02f}, "
                f"RA_q={best_val_trial.metrics['optuna/ra_query']:.02f}"
            )

            print(
                f"Best params (filtered) for robust_accuracy: \t{best_racc_trial.config} "
                f"@ CA={best_racc_trial.metrics['optuna/val_accuracy']:.02f}, "
                f"RA_min={best_racc_trial.metrics['optuna/robust_accuracy']:.02f}, "
                f"RA_t={best_racc_trial.metrics['optuna/ra_transfer']:.02f}, "
                f"RA_q={best_racc_trial.metrics['optuna/ra_query']:.02f}"
            )
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        x = [trial.metrics[objectives[0]] for trial in pareto_trials]
        y = [trial.metrics[objectives[1]] for trial in pareto_trials]

        df = pd.DataFrame({
            'x': x,
            'y': y
        })

        # Sort by x
        df_sorted = df.sort_values(by='x')
        sns.lineplot(ax=ax, data=df_sorted, x='x', y='y', marker='.', label=labels[idx], linewidth=5)

    ax.tick_params(axis='x', labelsize=fontsize - 4)
    ax.tick_params(axis='y', labelsize=fontsize - 4)

    ax.set_xlabel(r'$\mathsf{CA}$', fontsize=fontsize)
    ax.set_ylabel(r'$\min(\mathsf{RA}_\mathrm{T}, \mathsf{RA}_\mathrm{Q})$', fontsize=fontsize)

    plt.grid(True)

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=fontsize - 6, borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # or adjust right margin accordingly

    os.makedirs('data/plots/rq3', exist_ok=True)
    plt.savefig("data/plots/rq3/pareto.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
