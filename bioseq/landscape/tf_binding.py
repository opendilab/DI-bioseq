import os
import numpy as np
import pandas as pd
from typing import Union, Sequence

from bioseq.utils.registry import LANDSCAPE_REGISTRY
from .base_landscape import BaseLandscape

STARTS = [
    "GCTCGAGC",
    "GCGCGCGC",
    "TGCGCGCC",
    "ATATAGCC",
    "GTTTGGTA",
    "ATTATGTT",
    "CAGTTTTT",
    "AAAAATTT",
    "AAAAACGC",
    "GTTGTTTT",
    "TGCTTTTT",
    "AAAGATAG",
    "CCTTCTTT",
    "AAAGAGAG",
]


@LANDSCAPE_REGISTRY.register('tf_binding')
class TFBindingLandscape(BaseLandscape):

    _name = "tf_binding"

    def __init__(self, problem_name: str) -> None:
        super().__init__(landscape_type="DNAA")
        self._starts = STARTS

        tf_binding_data_dir = os.path.join(os.path.dirname(__file__), "data/tf_binding")
        self._all_prob_names = {}
        for fname in os.listdir(tf_binding_data_dir):
            pname = fname.replace("_8mers.txt", "")
            self._all_prob_names[pname] = fname

        if problem_name not in self._all_prob_names:
            raise ValueError("problem name {} not found".format(problem_name))

        landscape_file = os.path.join(tf_binding_data_dir, self._all_prob_names[problem_name])

        data = pd.read_csv(landscape_file, sep='\t')
        score = data['E-score']
        norm_score = (score - score.min()) / (score.max() - score.min())
        self._sequences = dict(zip(data['8-mer'], norm_score))
        self._sequences.update(zip(data['8-mer.1'], norm_score))

    def _get_score(self, sequences: Union[Sequence[str], np.ndarray]) -> np.ndarray:
        return np.array([self._sequences[seq] for seq in sequences])

    @property
    def starts(self):
        return self._starts
