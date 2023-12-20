from buffers import Buffer, PrioritizedReplayBuffer
from buffer_utils import Experience, Experience_weight_idx, LinearSchedule
from simple_config import (
    USE_PRIORITIZED_BUFFER,
    PRIORITIZED_REPLAY_ALPHA,
    PRIORITIZED_REPLAY_BETA0,
    PRIORITIZED_REPLAY_BETA_ITERS,
    PRIORITIZED_REPLAY_EPS,
    TRAIN_EPISODES,
)
from agent import Agent

from typing import Iterator, Tuple

from torch.utils.data.dataset import IterableDataset

import gym
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated
    with new experiences during training.

    Args:
        buffer: replay buffer
        batch_size: number of experiences to sample at a time
    """

    def __init__(
        self,
        buffer: Buffer,
        batch_size: int,
        env: gym.Env,
        actor_net: LightningModule,
        train_episodes: int,
        logger: WandbLogger,
    ) -> None:
        """
        Initialize buffer. If use_prioritized_buffer = 0 the normal buffer will
        be used, otherwise the prioritized experience replay buffer.
        """
        super(RLDataset).__init__()
        self.buffer = buffer
        self.batch_size = batch_size

        self.agent = Agent(env=env, buffer=buffer)
        self.total_reward = 0
        self.episode_reward = 0

        self.train_episodes = train_episodes
        self.episodes_done = 0

        self.actor_net = actor_net

        self.logger = logger
        self.step = 0

        if USE_PRIORITIZED_BUFFER:
            prioritized_replay_alpha = PRIORITIZED_REPLAY_ALPHA
            prioritized_replay_beta0 = PRIORITIZED_REPLAY_BETA0
            prioritized_replay_beta_iters = PRIORITIZED_REPLAY_BETA_ITERS
            prioritized_replay_eps = PRIORITIZED_REPLAY_EPS

            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = TRAIN_EPISODES
            # Create annealing schedule
            self.beta_schedule = LinearSchedule(
                prioritized_replay_beta_iters,
                initial_p=prioritized_replay_beta0,
                final_p=1.0,
            )

    def __iter__(self) -> Iterator[Tuple]:
        """
        Agent has to be implemented here. Super ugly but with Lightning and RL
        there are not non-wacky solutions.
        """
        self.step += 1
        # step through environment with agent
        reward, done = self.agent.play_step(self.actor_net)
        self.episode_reward += reward
        self.logger.log_metrics({"episode reward": self.episode_reward})
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self.logger.log_metrics({"reward": reward})
            self.logger.log_metrics({"total_reward": self.total_reward})
            self.episodes_done += 1
            self.logger.log_metrics({"episodes done": self.episodes_done})
        # if self.episodes_done > self.train_episodes:
        #     exit()

        if USE_PRIORITIZED_BUFFER:
            beta_value = self.beta_schedule.value(self.step)
            sampled_exps = self.buffer.sample(self.batch_size, beta_value)
        else:
            sampled_exps = self.buffer.sample(self.batch_size)

        yield sampled_exps
