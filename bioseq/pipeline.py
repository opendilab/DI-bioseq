import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import json


class Pipeline():

    def __init__(
            self,
            landscape,
            engine,
            rounds,
            score_per_round,
            predict_per_round,
            log_dir,
    ) -> None:
        self._landscape = landscape
        self._engine = engine
        self._name = (f"{self._landscape.name}-{self._engine.name}")

        self._rounds = rounds
        self._score_per_round = score_per_round
        self._predict_per_round = predict_per_round
        self._log_dir = log_dir
        self._log_file = None

    def run_pipeline(self, start_seq: str, run_id: int) -> pd.DataFrame:
        dir_path = os.path.join(self._log_dir, self._name.replace('-', '/', 1))
        os.makedirs(dir_path, exist_ok=True)
        self._log_file = dir_path + f"/res_{run_id}.csv"
        metadata = {
            "time_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "landscape": self._landscape.name,
            "engine": self._engine.name,
            "total_rounds": self._rounds,
            "score_per_round": self._score_per_round,
            "predict_per_round": self._predict_per_round,
        }

        start_data = pd.DataFrame(
            {
                'current_sequences': [start_seq],
                'predicted_scores': -1,
                'real_score': self._landscape.get_score([start_seq]),
                'round': 0,
                'predict_times': 0,
                'score_times': 1,
            }
        )
        current_data = start_data
        for r in range(self._rounds):
            start_time = time.time()
            data = current_data['current_sequences'].to_list()
            labels = current_data['real_score'].to_numpy()
            self._engine.update_model(data, labels)
            new_sequences, new_predictions = self._engine.generate_sequences(
                current_data['current_sequences'], self._score_per_round
            )
            real_score = self._landscape.get_score(new_sequences)
            current_data = current_data.append(
                pd.DataFrame(
                    {
                        'current_sequences': new_sequences,
                        'predicted_scores': new_predictions,
                        'real_score': real_score,
                        'round': r + 1,
                        'predict_times': self._engine.predict_times,
                        'score_times': self._landscape.score_times,
                    }
                )
            )
            end_time = time.time()
            self._record_info(current_data, metadata, r + 1, end_time - start_time)

        return current_data

    def reset(self) -> None:
        self._landscape.reset()
        self._engine.reset()

    def _record_info(self, data, metadata, current_round, time_cost):
        if self._log_file is not None:
            with open(self._log_file, 'w') as f:
                json.dump(metadata, f)
                f.write("\n")
                data.to_csv(f, index=False)

        print(f"round: {current_round}, top: {data['real_score'].max()}, " f"time: {time_cost}s")
