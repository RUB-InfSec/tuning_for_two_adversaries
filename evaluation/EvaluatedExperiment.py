import dataclasses
import logging

from pandas import DataFrame

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvaluatedExperiment:
    name: str = dataclasses.field(default_factory=str)
    config: dict = dataclasses.field(default_factory=dict)

    data: DataFrame = dataclasses.field(default_factory=DataFrame)
