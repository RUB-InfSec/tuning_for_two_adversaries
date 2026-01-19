# Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning

This repository contains the Python source code for the experiments of the paper "Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning
", to be published at AAAI 2026.

- [arXiV](https://arxiv.org/abs/2511.13654)
- Published version: TBD

## Setup

### Requirements

Create a fresh conda environment and activate it:

```bash
conda create --name tuning_artifact python=3.11.11 && conda activate tuning_artifact
```

Install requirements:

```bash
pip install -r requirements.txt
```

### Paths

The configs for the experiments of RQ1/RQ2 can be found in `experiments/rq12/...` and for RQ3 in `experiments/rq3/...`. To use the same dataset in the project folder and to prevent repetitive downloads (due to distribution through Ray), run the script `update_data_dir.sh` from the root of this project which will rewrite the files accordingly.

## Experiments

This framework enables model training with various hyperparameter configurations and evaluate this impact on robustness against black-box model evasion attacks. 

### Command-Line Arguments

```text
usage: main.py [-h] [--mode {create_config_rq12,run_rq12,run_rq3,evaluate_rq12,plot_rq12,plot_rq12_individual,plot_rq3}] [--experiment_name EXPERIMENT_NAME] [--gpus GPUS] [--seed SEED]
               [--opt_args [OPT_ARGS ...]] [--randomize] [--force]

options:
  -h, --help            show this help message and exit
  --mode {create_config_rq12,run_rq12,run_rq3,evaluate_rq12,plot_rq12,plot_rq3}
                        Mode to run.
  --experiment_name EXPERIMENT_NAME
                        Experimental configuration to use for the experiment.
  --gpus GPUS           Comma separated list of GPUs to use. Sorted by PCI_BUS_ID.
  --seed SEED           Global seed for seeding all other seeds.
  --opt_args [OPT_ARGS ...]
                        Optional arguments.
  --randomize           Triggers three randomized repetitions of an experiment.
  --force               Overrides already evaluated experiments.
```

To recreate the experiments, the following commands need to be executed from the root of the project. If path-related errors occur, please preprend `PYTHONPATH=$(pwd)` before the `python` command.

### RQ1/RQ2:
- Run all parameter configurations for a given ML instance for three rounds with
  `python main.py --mode run_rq12 --randomize --experiment_name experiments/rq12/cifar10/[lr/bs/mom/wd]/{config}.yaml` where `{config}` refers to
  a config file in the given directory.
- The trained models are evaluated for clean accuracy and robust accuracies with
  `python main.py --mode evaluate_rq12 --randomize --experiment_name experiments/rq12/cifar10/[lr/bs/mom/wd]/{config}.yaml`.
- The plots of Figure 2 can be created with `python main.py --mode plot_rq12` and are saved in `data/plots/rq12`. This only works when all three runs have been trained and evaluated as this specific plot averages over multiple dimensions.


### RQ3:
- The hyperparameter search with NSGA-II can be executed for each ML instantiation with
  `python main.py --mode run_rq3 --experiment_name experiments/rq3/{config}.yaml`.
- The Pareto plot for Figure 3 can be plotted with `python main.py --mode plot_rq3` and is saved in `data/plots/rq3`. This will also output the
  best hyperparameters for each ML instance on the console.

### Optional:
- Create a custom config by editing the template in `utils/generate_config.py` and creating a YAML file with `python main.py --mode create_config_rq12 --experiment_name {config}`
- Plotting the accuracies of an individual experiment can be done with: `python main.py --mode plot_rq12_individual --experiment_name experiments/rq12/cifar10/[lr/bs/mom/wd]/{config}.yaml`. 


## Citation
```
@misc{zimmer2025tuningadversariesenhancingrobustness,
      title={Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning}, 
      author={Pascal Zimmer and Ghassan Karame},
      year={2025},
      eprint={2511.13654},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.13654}, 
}
```
## Contact

Feel free to contact the first author via the e-mail provided on the publication if you have any questions.