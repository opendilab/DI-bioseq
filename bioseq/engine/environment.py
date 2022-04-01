import numpy as np
from typing import Any, Dict, Optional

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElementInfo
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register("bioseq_mutative")
class MutativeEnv(BaseEnv):

    reward_space = None

    def __init__(self, model, encoder, codebook, start_seq) -> None:
        self._model = model
        self._encoder = encoder
        self._codebook = codebook
        self._cur_seq = start_seq

        self._episode_seqs = set()
        self._last_fitness = -float("inf")
        self._lam = 0.1

    def reset(self, start_seq: Optional[str] = None) -> Any:
        self._total_reward = 0
        if start_seq is not None:
            self._cur_seq = start_seq
        self._cur_encoding = self._encoder.encode([self._cur_seq])[0]
        cur_fitness = self._model.predict(np.array([self._cur_encoding]))[0]
        self._last_fitness = cur_fitness
        self._episode_seqs.clear()
        self._episode_seqs.add(self._cur_seq)
        return to_ndarray(self._cur_encoding.copy().reshape(-1), dtype=np.float32)

    def step(self, action: int) -> BaseEnvTimestep:
        pos = action // len(self._codebook)
        res = action % len(self._codebook)
        if self._cur_encoding[pos, res] == 1:
            obs = self._cur_encoding.copy().reshape(-1)
            reward = -1
            done = True
            info = {'seq': self._cur_seq, 'fitness': self._last_fitness}
        else:
            self._cur_encoding[pos] = 0
            self._cur_encoding[pos, res] = 1
            self._cur_seq = self._encoder.decode([self._cur_encoding])[0]
            fitness = self._model.predict(np.array([self._cur_encoding]))[0]
            obs = self._cur_encoding.copy().reshape(-1)
            info = {'seq': self._cur_seq, 'fitness': fitness}
            if self._cur_seq in self._episode_seqs:
                reward = -1
                done = True
            else:
                self._episode_seqs.add(self._cur_seq)
                # if fitness < self._last_fitness:
                #     reward = 0
                reward = fitness
                done = False
            self._last_fitness = fitness
        self._total_reward += reward
        if done:
            info['final_eval_reward'] = self._total_reward
        return BaseEnvTimestep(to_ndarray(obs, dtype=np.float32), to_ndarray(reward, dtype=np.float32), done, info)

    def info(self) -> BaseEnvInfo:
        obs_space = EnvElementInfo(
            (len(self._cur_seq) * len(self._codebook), ),
            {
                'min': 0,
                'max': 1,
                'dtype': np.float32,
            },
        )
        act_space = EnvElementInfo(
            (1, ),
            {
                'min': 0,
                'max': len(self._cur_seq) * len(self._codebook),
                'dtype': int,
            },
        )
        rew_space = EnvElementInfo(
            (1, ),
            {
                'min': 0.0,
                'max': 1.0,
            },
        )
        return BaseEnvInfo(
            agent_num=1, obs_space=obs_space, act_space=act_space, rew_space=rew_space, use_wrappers=False
        )

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self):
        self._episode_seqs.clear()

    def __repr__(self) -> str:
        return "Bioseq MutativeEnv"
