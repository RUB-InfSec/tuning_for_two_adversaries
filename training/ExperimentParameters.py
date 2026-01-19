import dataclasses
import inspect
import os
from collections import defaultdict
from typing import Iterator

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.nn import Parameter
from torchvision.transforms.v2 import Transform, Compose

from models.cifar10_models import ResNet18, MobileNetV2
from training.CustomDataSet import SplitByIndexDataset


@dataclasses.dataclass
class NetworkParameters:
    architecture_name: str = ""
    initialization_seed: int = 24
    dropout_rate: float = 0.0


@dataclasses.dataclass
class DataParameters:
    dataset: str = ""
    data_dir: str = os.path.abspath("data/")
    validation_split: float = 0.2
    data_split: str = "None"
    batch_size: int = 0
    node_id: int = 0
    node_count: int = 1
    data_split_params: list = dataclasses.field(default_factory=list)  # i.e., dirichlet alpha
    splitting_seed: int = 123
    batching_seed: int = 456

    train_transforms: Transform = dataclasses.field(default_factory=Transform)
    test_transforms: Transform = dataclasses.field(default_factory=Transform)


@dataclasses.dataclass
class OptimizerParameters:
    optimizer_name: str = ""
    learning_rate: float = 0.0
    momentum: float = 0.0
    weight_decay: float = 0.0


@dataclasses.dataclass
class SchedulerParameters:
    scheduler_name: str = ""
    scheduler_params: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TrainingParameters:
    epochs: int = 0
    early_stopping: bool = False


@dataclasses.dataclass
class EvaluationParameters:
    ensembles_only: bool = False
    ensemble_keys: list = dataclasses.field(default_factory=list)
    transfer_file: str = ""


@dataclasses.dataclass
class ExperimentParameters:
    global_seed: int = 42
    train_git_hash: str = ""

    evaluation_parameters: EvaluationParameters = dataclasses.field(default_factory=EvaluationParameters)
    training_parameters: TrainingParameters = dataclasses.field(default_factory=TrainingParameters)
    data_parameters: DataParameters = dataclasses.field(default_factory=DataParameters)
    network_parameters: NetworkParameters = dataclasses.field(default_factory=NetworkParameters)
    optimizer_parameters: OptimizerParameters = dataclasses.field(default_factory=OptimizerParameters)
    scheduler_parameters: SchedulerParameters = dataclasses.field(default_factory=SchedulerParameters)

    def obj2dict(self, obj):
        if isinstance(obj, dict):
            return {k: self.obj2dict(v) for k, v in obj.items()}
        elif not isinstance(obj, str) and hasattr(obj, "__iter__"):
            return [self.obj2dict(v) for v in obj]
        elif isinstance(obj, torch.dtype):
            return obj.__repr__()
        elif isinstance(obj, Compose):
            return [self.obj2dict(v) for v in obj.transforms]
        elif isinstance(obj, Transform):
            parm = {}
            init_signature = inspect.signature(obj.__class__.__init__)

            # Extract default parameters
            defaults = {
                k: v.default
                for k, v in init_signature.parameters.items()
                if k != 'self'  # Ignore 'self' parameter
            }

            # Extract required parameters (those without a default value)
            required_params = {k for k, v in init_signature.parameters.items() if
                               v.default == inspect.Parameter.empty and k != 'self'}

            # Get the values set in the instance
            instance_params = {k: getattr(obj, k, None) for k in defaults.keys()}

            # Find changed parameters (either differing from default or required parameters)
            changed_params = {
                k: v for k, v in instance_params.items()
                if k in required_params or (k in defaults and defaults[k] != v)
            }

            for k, v in changed_params.items():
                parm.update({k: self.obj2dict(v)})

            return {'name': obj.__class__.__name__, 'params': parm}
        elif hasattr(obj, "__dict__"):
            return {
                k: self.obj2dict(v)
                for k, v in obj.__dict__.items()
                # if not callable(v) and not k.startswith('_')
                if not k.startswith('_')
            }
        else:
            return obj

    @staticmethod
    def from_dict(input_dict):
        obj = ExperimentParameters()

        def get_cls_instance(name: str):
            module_name, attr_name = name.rsplit(".", 1)

            # Import the module dynamically (if not already imported)
            module = globals().get(module_name)
            if module is None:
                module = __import__(module_name)

            # Get the attribute from the module
            dtype = getattr(module, attr_name)

            return dtype

        for key, value in input_dict.items():
            if key == 'metrics':
                # skip the metrics key that might have been merged into it throughout an experiment
                continue
            elif key == 'data_parameters':
                date_parameters = DataParameters()
                for inner_key, inner_value in input_dict['data_parameters'].items():
                    if inner_key.endswith('_transforms'):
                        trans = []
                        possible_transforms = dict(inspect.getmembers(transforms))
                        for transfo in input_dict['data_parameters'][inner_key]:
                            if transfo['name'] in possible_transforms:
                                cls = possible_transforms[transfo['name']]
                                parma = transfo['params']
                                if len(parma) > 0:
                                    try:
                                        trans.append(cls(**parma))
                                    except ValueError:
                                        eval_params = {k: get_cls_instance(v) if isinstance(v, str) and "." in v else v
                                                       for k, v in parma.items()}
                                        trans.append(cls(**eval_params))
                                else:
                                    trans.append(cls())
                            else:
                                raise ValueError("Invalid transformation found.")

                        setattr(date_parameters, inner_key, transforms.Compose(transforms=trans))
                    else:
                        setattr(date_parameters, inner_key, inner_value)
                setattr(obj, key, date_parameters)
            elif key == 'network_parameters':
                network_parameters = NetworkParameters()
                for inner_key, inner_value in input_dict['network_parameters'].items():
                    setattr(network_parameters, inner_key, inner_value)
                setattr(obj, key, network_parameters)
            elif key == 'optimizer_parameters':
                optimizer_parameters = OptimizerParameters()
                for inner_key, inner_value in input_dict['optimizer_parameters'].items():
                    setattr(optimizer_parameters, inner_key, inner_value)
                setattr(obj, key, optimizer_parameters)
            elif key == 'scheduler_parameters':
                scheduler_parameters = SchedulerParameters()
                for inner_key, inner_value in input_dict['scheduler_parameters'].items():
                    setattr(scheduler_parameters, inner_key, inner_value)
                setattr(obj, key, scheduler_parameters)
            elif key == 'evaluation_parameters':
                evaluation_parameters = EvaluationParameters()
                for inner_key, inner_value in input_dict['evaluation_parameters'].items():
                    setattr(evaluation_parameters, inner_key, inner_value)
                setattr(obj, key, evaluation_parameters)
            elif key == 'training_parameters':
                training_parameters = TrainingParameters()
                for inner_key, inner_value in input_dict['training_parameters'].items():
                    setattr(training_parameters, inner_key, inner_value)
                setattr(obj, key, training_parameters)
            else:
                setattr(obj, key, value)

        return obj

    def to_dict(self):
        return self.obj2dict(self)

    def get_dataset(self, train=False) -> torch.utils.data.Dataset:

        dataset_name = self.data_parameters.dataset
        datasplit = self.data_parameters.data_split

        nmodels = self.data_parameters.node_count
        nid = self.data_parameters.node_id

        assert nid < nmodels

        if not train:
            transform = self.data_parameters.test_transforms
        else:
            transform = self.data_parameters.train_transforms

        if dataset_name == "cifar10":
            the_set = torchvision.datasets.CIFAR10(root=self.data_parameters.data_dir, train=train,
                                                   download=True, transform=transform)
        else:
            raise NotImplementedError("Unsupported dataset.")

        if datasplit == "None" or not train or self.data_parameters.node_count == 1:
            return the_set
        elif datasplit in ["uniform", "dirichlet"]:
            rng = np.random.default_rng(seed=self.data_parameters.splitting_seed)

            dataset_classes = {}
            for ind, label in enumerate(the_set):
                label = label[1]
                if label in dataset_classes:
                    dataset_classes[label].append(ind)
                else:
                    dataset_classes[label] = [ind]

            per_node_indices = defaultdict(list)
            no_classes = len(dataset_classes.keys())

            for n in range(no_classes):
                rng.shuffle(dataset_classes[n])
                class_size = len(dataset_classes[n])
                if datasplit == "uniform":
                    sampled_probabilities = class_size * np.array(nmodels * [1 / nmodels])
                else:
                    alpha = self.data_parameters.data_split_params[0]
                    sampled_probabilities = class_size * rng.dirichlet(np.array(nmodels * [alpha]))

                for node in range(nmodels):
                    no_imgs = int(round(sampled_probabilities[node]))
                    sampled_list = dataset_classes[n][:min(len(dataset_classes[n]), no_imgs)]
                    per_node_indices[node].extend(sampled_list)
                    dataset_classes[n] = dataset_classes[n][min(len(dataset_classes[n]), no_imgs):]

            split_dataset = SplitByIndexDataset(per_node_indices[nid], the_set)

        return split_dataset

    def get_model(self) -> torch.nn.Module:

        architecture_name = self.network_parameters.architecture_name
        dataset = self.data_parameters.dataset

        if dataset == "cifar10":
            match architecture_name:
                case "resnet18":
                    mod = ResNet18()
                case "mobilenet_v2":
                    mod = MobileNetV2()
                case _:
                    raise NotImplementedError("Unsupported architecture.")
        else:
            raise NotImplementedError("Unsupported dataset.")

        return mod

    def get_optimizer(self, model_params: Iterator[Parameter]) -> torch.optim.Optimizer:
        optim_name = self.optimizer_parameters.optimizer_name
        learning_rate = self.optimizer_parameters.learning_rate
        momentum = self.optimizer_parameters.momentum
        weight_decay = self.optimizer_parameters.weight_decay

        match optim_name:
            case "SGD":
                return torch.optim.SGD(model_params, lr=learning_rate, momentum=0.0, weight_decay=weight_decay)
            case "SGD-M":
                return torch.optim.SGD(model_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            case "SGD-N":
                return torch.optim.SGD(model_params, lr=learning_rate, momentum=momentum, nesterov=True,
                                       weight_decay=weight_decay)
            case _:
                raise NotImplementedError("Unsupported optimizer.")

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler_name = self.scheduler_parameters.scheduler_name

        match scheduler_name:
            case "CosineAnnealingLR":
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.training_parameters.epochs)
            case "None":
                return None
            case _:
                raise NotImplementedError("Unsupported scheduler.")
