from typing import List, Sequence
import numpy as np

from bioseq.utils.registry import ENCODER_REGISTRY
from .base_encoder import BaseEncoder


@ENCODER_REGISTRY.register('onehot')
class OneHotEncoder(BaseEncoder):
    name = "onehot"

    def __init__(self, codebook: Sequence[str]) -> None:
        self._codebook = codebook

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        return np.array([self._string_to_onehot(seq) for seq in sequences])

    def decode(self, array: np.ndarray) -> List[str]:
        return [self._onehot_to_string(item) for item in array]

    def _string_to_onehot(self, seq):
        out = np.zeros((len(seq), len(self._codebook)))
        for i in range(len(seq)):
            out[i, self._codebook.index(seq[i])] = 1
        return out

    def _onehot_to_string(self, onehot):
        residue_idxs = np.argmax(onehot, axis=1)
        return "".join([self._codebook[idx] for idx in residue_idxs])
