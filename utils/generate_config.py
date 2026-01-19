import torch
import torchvision.transforms.v2 as transforms
from ray import tune

from training.ExperimentParameters import DataParameters, SchedulerParameters, OptimizerParameters, NetworkParameters, \
    TrainingParameters, ExperimentParameters, EvaluationParameters


def cifar10_query_budgets():
    return [500]


def cifar10_transforms():
    test_transform = transforms.Compose(
        [transforms.ToImage(),
         transforms.ToDtype(torch.float32, scale=True),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_transform = transforms.Compose(
        [transforms.ToImage(),
         transforms.ToDtype(torch.uint8, scale=True),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(size=(32, 32), padding=4),
         transforms.ToDtype(torch.float32, scale=True),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    return test_transform, train_transform


def generate_experiment_config() -> dict:
    # initialize default configuration
    test_transform, train_transform = cifar10_transforms()

    # --- edit after this line ---
    tp = TrainingParameters(epochs=200)
    nwp = NetworkParameters(architecture_name="resnet18")
    dp = DataParameters(dataset="cifar10", data_split="uniform", data_split_params=[0.9], batch_size=128, node_count=1,
                        node_id=0, test_transforms=test_transform, train_transforms=train_transform)
    op = OptimizerParameters(optimizer_name="SGD-M", learning_rate=0.1, momentum=0.9, weight_decay=5e-4)
    sp = SchedulerParameters(scheduler_name="CosineAnnealingLR", scheduler_params=[])

    # ---
    # centralized and deep ensemble instances
    # ep = EvaluationParameters(ensembles_only=True, ensemble_keys=["optimizer_parameters.learning_rate"], transfer_file="cifar10_various_architectures.yaml")

    # distributed ML instances
    ep = EvaluationParameters(ensembles_only=True, ensemble_keys=["-data_parameters.node_id", "filtered_parameter_wise"], transfer_file="cifar10_various_architectures.yaml")

    params = ExperimentParameters(evaluation_parameters=ep, training_parameters=tp, data_parameters=dp,
                                  network_parameters=nwp,
                                  optimizer_parameters=op, scheduler_parameters=sp)

    params_dict = params.to_dict()

    params_dict["global_seed"] = tune.randint(0, 1000)
    params_dict["data_parameters"]["batching_seed"] = tune.randint(0, 1000)
    params_dict["network_parameters"]["initialization_seed"] = tune.randint(0, 1000)

    # deep ensembles: full data, but different initialization
    # params_dict["network_parameters"]["initialization_seed"] = tune.grid_search([1, 2, 3])
    # params_dict["network_parameters"]["initialization_seed"] = tune.grid_search([4, 5, 6, 7, 8])
    # params_dict["network_parameters"]["initialization_seed"] = tune.grid_search([9, 10, 11, 12, 13, 14, 15])

    # distributed ML: uniform/dirichlet split
    params_dict["data_parameters"]["node_count"] = 7
    params_dict["data_parameters"]["data_split"] = "dirichlet"
    params_dict["data_parameters"]["node_id"] = tune.grid_search(
        [i for i in range(params_dict["data_parameters"]["node_count"])])

    # specify hyperparameter ranges
    params_dict["optimizer_parameters"]["learning_rate"] = tune.grid_search(
        [0.0001, 0.0002, 0.0004, 0.0008, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4])
    # params_dict["optimizer_parameters"]["momentum"] = tune.grid_search([0.8, 0.85, 0.89, 0.9, 0.95, 0.99])
    # params_dict["optimizer_parameters"]["weight_decay"] = tune.grid_search(
    #     [1e-6, 2e-6, 4e-6, 8e-6, 1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 2e-4, 4e-4, 5e-4, 8e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1e-2])
    # params_dict["data_parameters"]["batch_size"] = tune.grid_search([32, 64, 128, 256, 512, 1024, 2048])

    return params_dict
