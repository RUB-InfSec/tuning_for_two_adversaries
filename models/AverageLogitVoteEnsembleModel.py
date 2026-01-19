from typing import List

import torch

from models.EnsembleModel import EnsembleModel
from models.WeakLearner import WeakLearner


class AverageLogitVoteEnsembleModel(EnsembleModel):

    def __init__(self, models: List[WeakLearner]):
        super().__init__(models)

    def forward(self, x):
        outs = [mod(x) for mod in self.models]
        return torch.stack(outs).mean(dim=0)
