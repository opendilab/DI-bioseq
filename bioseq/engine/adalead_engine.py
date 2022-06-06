import numpy as np
from easydict import EasyDict
from typing import Any, Sequence, Tuple

from bioseq.utils.registry import ENGINE_REGISTRY
from .base_engine import BaseEngine


@ENGINE_REGISTRY.register('adalead')
class AdaleadEngine(BaseEngine):
    _name = "adalead"

    def __init__(
            self,
            model: "BaseModel",
            encoder: Any,
            codebook: str,
            predict_num: int,
            seq_len: int,
            threshold: float = 0.05,
            recombine_rate: float = 0,
            mu: int = 1,
            rho: int = 0,
            model_batch_size: int = 20,
    ) -> None:
        super().__init__(model, encoder, codebook, predict_num)
        self._threshold = threshold
        self._recombine_rate = recombine_rate
        self._mu = mu
        self._rho = rho
        self._model_batch_size = model_batch_size

    def generate_sequences(self, old_sequences: Sequence[str], scores: Sequence[float],
                           sequence_num: int) -> Tuple[np.ndarray, np.ndarray]:
        top_score = scores.max()
        top_indexes = scores >= top_score * (1 - np.sign(top_score) * self._threshold)
        parents = np.resize(np.array(old_sequences)[top_indexes], sequence_num)
        sequences = {}
        previous_predict_times = self._model.predict_times
        while self._model.predict_times - previous_predict_times < self._predict_num:
            for i in range(self._rho):
                parents = self._recombine_population(parents)
            for i in range(0, len(parents), self._model_batch_size):
                roots = parents[i:i + self._model_batch_size]
                encodings = self._encoder.encode(roots)
                root_prediction = self._model.predict(encodings, ignore_count=True)
                nodes = list(enumerate(roots))
                while (len(nodes) > 0 and len(sequences) + self._model_batch_size < self._predict_num):
                    child_idxs = []
                    children = []
                    while len(children) < len(nodes):
                        idx, node = nodes[len(children) - 1]
                        child = self._generate_random_mutant(node)
                        if (child not in old_sequences and child not in sequences):
                            child_idxs.append(idx)
                            children.append(child)
                    encodings = self._encoder.encode(children)
                    predictions = self._model.predict(encodings)
                    sequences.update(zip(children, predictions))
                    nodes = []
                    for idx, child, fitness in zip(child_idxs, children, predictions):
                        if fitness >= root_prediction[idx]:
                            nodes.append((idx, child))
        if len(sequences) == 0:
            raise ValueError("No sequences generated.")
        new_sequences = np.array(list(sequences.keys()))
        new_values = np.array(list(sequences.values()))
        max_indexs = np.argsort(new_values)[:-sequence_num - 1:-1]
        new_sequences = np.array(new_sequences)[max_indexs]
        new_predictions = new_values[max_indexs]
        return new_sequences, new_predictions

    def _recombine_population(self, seqs):
        # If only one member of population, can't do any recombining
        if len(seqs) == 1:
            return seqs

        np.random.shuffle(seqs)
        res = []
        for i in range(0, len(seqs) - 1, 2):
            str1 = []
            str2 = []
            switch = False
            for ind in range(len(seqs[i])):
                if np.random.rand() < self._recombine_rate:
                    switch = not switch
                # putting together recombinants
                if switch:
                    str1.append(seqs[i][ind])
                    str2.append(seqs[i + 1][ind])
                else:
                    str1.append(seqs[i][ind])
                    str2.append(seqs[i + 1][ind])
            res.append("".join(str1))
            res.append("".join(str2))
        return res

    def _generate_random_mutant(self, seq):
        res = ""
        for s in seq:
            if np.random.rand() < self._mu / len(seq):
                res += np.random.choice(list(self._codebook))
            else:
                res += s
        return res

    def update_model(self, data: np.ndarray, labels: np.ndarray) -> None:
        encodings = self._encoder.encode(data)
        self._model.train(encodings, labels)
