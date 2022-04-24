import numpy as np
from easydict import EasyDict
from typing import Any, Sequence, Tuple

from ding.policy import PPOOffPolicy
from ding.envs import AsyncSubprocessEnvManager, BaseEnvManager
from ding.worker import BaseLearner, SampleSerialCollector, NaiveReplayBuffer
from ding.config import compile_config
from bioseq.engine.environment import MutativeEnv
from bioseq.utils.registry import ENGINE_REGISTRY
from .base_engine import BaseEngine

ppo_config = dict(
    exp_name="bio_rl",
    env=dict(
        env_num=8,
        manager=dict(
            #context='spawn',
            shared_memory=False,
        ),
        n_evaluator_episode=0,
        stop_value=1.0,
    ),
    policy=dict(
        cuda=True,
        model=dict(),
        learn=dict(
            batch_size=64,
            update_per_collect=20,
            learner=dict(
                hook=dict(
                    log_show_after_iter=1e8,
                    save_ckpt_after_iter=1e8,
                ),
            ),
        ),
        collect=dict(
            n_sample=320,
            collector=dict(
                collect_print_freq=1e8,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=10000,
                periodic_thruput_seconds=1e8,
            ),
        ),
    ),
)
rl_cfg = EasyDict(ppo_config)


def get_env_fn(model, encoder, codebook, start_seq):

    def _env_fn():
        env = MutativeEnv(model, encoder, codebook, start_seq)
        return env

    return _env_fn


@ENGINE_REGISTRY.register("ppo_offpolicy")
class PPOEngine(BaseEngine):

    _name = "ppo_offpolicy"

    def __init__(
            self,
            model: "BaseModel",  # noqa
            encoder: "BaseEncoder",  # noqa
            codebook: str,
            predict_num: int,
            seq_len: int,
    ) -> None:
        super().__init__(model, encoder, codebook, predict_num)
        self._seq_len = seq_len

        self._rl_cfg = compile_config(
            cfg=rl_cfg,
            env_manager=AsyncSubprocessEnvManager,
            policy=PPOOffPolicy,
            learner=BaseLearner,
            collector=SampleSerialCollector,
            buffer=NaiveReplayBuffer,
            save_cfg=False,
        )
        act_shape = seq_len * len(self._codebook)
        obs_shape = seq_len * len(self._codebook)
        self._rl_cfg.policy.model.action_shape = act_shape
        self._rl_cfg.policy.model.obs_shape = obs_shape

    def generate_sequences(self, old_sequences: Sequence[str], sequence_num: int) -> Tuple[np.ndarray, np.ndarray]:
        env_num = self._rl_cfg.env.env_num
        env = AsyncSubprocessEnvManager(
            env_fn=[
                get_env_fn(self._model, self._encoder, self._codebook, np.random.choice(old_sequences))
                for i in range(env_num)
            ],
            cfg=self._rl_cfg.env.manager
        )
        collector = SampleSerialCollector(
            self._rl_cfg.policy.collect.collector, env, self._policy.collect_mode, exp_name=self._rl_cfg.exp_name
        )

        sequences = {}
        data_num = 0
        while data_num < self._predict_num:
            new_data = collector.collect(self._rl_cfg.policy.collect.n_sample, train_iter=self._learner.train_iter)
            data_num += len(new_data)
            for data in new_data:
                code = data['obs'].cpu().reshape(1, self._seq_len, -1)
                seq = self._encoder.decode(np.array(code))[0]
                fitness = self._model.predict(np.array(code))[0]
                sequences.update({seq: fitness})
            self._replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            for i in range(self._rl_cfg.policy.learn.update_per_collect):
                train_data = self._replay_buffer.sample(self._rl_cfg.policy.learn.batch_size, self._learner.train_iter)
                self._learner.train(train_data, collector.envstep)

        sequences = {seq: fitness for seq, fitness in sequences.items() if seq not in old_sequences}
        new_sequences = np.array(list(sequences.keys()))
        new_values = np.array(list(sequences.values()))
        max_indexs = np.argsort(new_values)[:-sequence_num - 1:-1]
        new_sequences = np.array(new_sequences)[max_indexs]
        new_predictions = new_values[max_indexs]
        return new_sequences, new_predictions

    def update_model(self, data: np.ndarray, labels: np.ndarray) -> None:
        encodings = self._encoder.encode(data)
        self._model.train(encodings, labels)

    def reset(self) -> None:
        super().reset()
        self._policy = PPOOffPolicy(self._rl_cfg.policy)
        self._learner = BaseLearner(
            self._rl_cfg.policy.learn.learner, self._policy.learn_mode, exp_name=self._rl_cfg.exp_name
        )
        self._replay_buffer = NaiveReplayBuffer(self._rl_cfg.policy.other.replay_buffer, exp_name=self._rl_cfg.exp_name)
