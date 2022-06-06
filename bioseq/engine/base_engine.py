import numpy as np
from abc import abstractmethod
from typing import Any, Sequence, Tuple

from bioseq.utils.registry import ENGINE_REGISTRY


class BaseEngine():

    def __init__(
            self,
            model: "BaseModel",  # noqa
            encoder: "BaseEncoder",  # noqa
            codebook: str,
            predict_num: int,
    ) -> None:
        self._model = model
        self._encoder = encoder
        self._codebook = codebook
        self._predict_num = predict_num

    @abstractmethod
    def generate_sequences(self, old_sequences: Sequence[str], scores: Sequence[float],
                           sequence_num: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def update_model(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        self._model.reset()

    @property
    def name(self):
        return f"{self._name}-{self._model.name}-{self._encoder.name}"

    @property
    def predict_times(self):
        return self._model.predict_times


def create_engine(name, *args, **kwargs):
    return ENGINE_REGISTRY.build(name, *args, **kwargs)
