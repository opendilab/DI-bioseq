import numpy as np
from abc import abstractmethod
from typing import Sequence, Union

from bioseq.utils.registry import LANDSCAPE_REGISTRY

CODEBOOK_DICT = dict(
    AAS="ILVAGMFYWEDQNHCRKSTP",
    RNAA="UGCA",
    DNAA="TGCA",
)


class BaseLandscape():

    _name = None

    def __init__(self, landscape_type: str) -> None:
        self._score_times = 0
        self._codebook = CODEBOOK_DICT[landscape_type]

    @abstractmethod
    def _get_score(self, sequences: Union[Sequence[str], np.ndarray]) -> np.ndarray:
        pass

    def get_score(self, sequences: Union[Sequence[str], np.ndarray]) -> np.ndarray:
        self._score_times += len(sequences)
        return self._get_score(sequences)

    def reset(self) -> None:
        self._score_times = 0

    @property
    def codebook(self):
        return self._codebook

    @property
    def seq_len(self):
        return len(self.starts[0])

    @property
    def score_times(self):
        return self._score_times

    @property
    def name(self):
        return self._name


def create_landscape(name, *args, **kwargs):
    if ':' not in name:
        return LANDSCAPE_REGISTRY.build(name, *args, **kwargs)
    else:
        return LANDSCAPE_REGISTRY.build(*name.split(':', 1), *args, **kwargs)
