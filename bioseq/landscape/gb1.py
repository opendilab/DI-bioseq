import os
import pandas as pd
import numpy as np
from typing import Sequence

from bioseq.utils.registry import LANDSCAPE_REGISTRY
from .base_landscape import BaseLandscape

ONLY_MEASURED_STARTS = [
    "FWRA",
    "FWWA",
    "FWSA",
    "FEAA",
    "KWAY",
    "FTAY",
    "FCWA",
    "MWSA",
    "FHSA",
    "KLNA",
    "SNVA",
    "IYAN",
    "WEGA",
    "LLLA",
    "EVWL",
    "LFDI",
    "VYFL",
    "YFCI",
    "VYVG",
]

WITH_IMPUTED_STARTS = [
    "AHNA",
    "CHCA",
    "NHCA",
    "ATRA",
    "KAHC",
    "AHHK",
    "CRCA",
    "AHGC",
    "WHRH",
    "GSCQ",
    "AGFR",
    "YYCS",
    "AMLG",
    "SIDW",
    "CMPW",
    "VRFM",
    "KVGF",
    "MRGM",
]

SEARCH_SPACE = "V39,D40,G41,V54"


@LANDSCAPE_REGISTRY.register('gb1')
class GB1Landscape(BaseLandscape):
    _name = "gb1"

    def __init__(self, problem_name: str = "with_imputed") -> None:
        super().__init__(landscape_type="AAS")
        self._starts = WITH_IMPUTED_STARTS

        measured_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/gb1/elife-16965-supp1-v4.csv"))
        measured_data = measured_data[["Variants", "Fitness"]]
        if problem_name == "with_imputed":
            imputed_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/gb1/elife-16965-supp2-v4.csv"))
            imputed_data.columns = ["Variants", "Fitness"]
            data = pd.concat([measured_data, imputed_data])
            self._starts = WITH_IMPUTED_STARTS
        elif problem_name == "only_measured":
            data = measured_data
            self._starts = ONLY_MEASURED_STARTS
        else:
            raise ValueError("problem name {} not found".format(problem_name))

        score = data["Fitness"]
        norm_score = (score - score.min()) / (score.max() - score.min())
        self._sequences = dict(zip(data["Variants"], norm_score))

    def _get_score(self, sequence: Sequence[str]) -> np.ndarray:
        return np.array([self._sequences[seq] for seq in sequence], dtype=np.float32)

    @property
    def starts(self):
        return self._starts
