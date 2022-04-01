from typing import List, Sequence
import numpy as np
from abc import abstractmethod

from bioseq.utils.registry import ENCODER_REGISTRY


class BaseEncoder():

    @abstractmethod
    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        pass


def create_encoder(name, *args, **kwargs):
    return ENCODER_REGISTRY.build(name, *args, **kwargs)
