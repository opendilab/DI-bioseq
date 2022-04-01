import numpy as np
from typing import Any, Sequence, Tuple

from bioseq.utils.registry import ENGINE_REGISTRY
from .base_engine import BaseEngine


@ENGINE_REGISTRY.register("random")
class RandomEngine(BaseEngine):

    _name = "random"

    def __init__(
            self,
            model: "BaseModel",  # noqa
            encoder: Any,
            codebook: str,
            predict_num: int,
            seq_len: int,
            mu: float = 1,
    ) -> None:
        super().__init__(model, encoder, codebook, predict_num)
        self._mu = mu

    def generate_sequences(self, old_sequences: Sequence[str], sequence_num: int) -> Tuple[np.ndarray, np.ndarray]:
        new_sequences = set()
        while len(new_sequences) < self._predict_num:
            seq = np.random.choice(old_sequences)
            new_seq = self._generate_random_mutant(seq)
            if new_seq not in old_sequences:
                new_sequences.add(new_seq)
        new_sequences = sorted(new_sequences)
        encodings = self._encoder.encode(new_sequences)
        predictions = self._model.predict(encodings)
        max_indexs = np.argsort(predictions)[:-sequence_num - 1:-1]
        new_sequences = np.array(new_sequences)[max_indexs]
        new_predictions = predictions[max_indexs]
        return new_sequences, new_predictions

    def update_model(self, data: np.ndarray, labels: np.ndarray) -> None:
        encodings = self._encoder.encode(data)
        self._model.train(encodings, labels)

    def _generate_random_mutant(self, seq):
        res = ""
        for s in seq:
            if np.random.rand() < self._mu:
                res += np.random.choice(list(self._codebook))
            else:
                res += s
        return res
