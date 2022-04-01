import numpy as np
from typing import Any, Union, Sequence
from abc import abstractmethod

from bioseq.utils.registry import MODEL_REGISTRY


class BaseModel():

    _name = None

    def __init__(self) -> None:
        self._predict_times = 0

    @abstractmethod
    def train(self, sequences: Union[Sequence[str], np.ndarray], label: Sequence[Any]):
        pass

    @abstractmethod
    def _fit(self, sequences: Union[Sequence[str], np.ndarray], label: Sequence[Any]):
        pass

    def predict(self, sequences: Union[Sequence[str], np.ndarray]) -> np.array:
        self._predict_times += len(sequences)
        return self._fit(sequences)

    def reset(self) -> None:
        self._predict_times = 0

    @property
    def predict_times(self) -> int:
        return self._predict_times

    @property
    def name(self):
        return self._name


def create_model(name, *args, **kwargs):
    return MODEL_REGISTRY.build(name, *args, **kwargs)
