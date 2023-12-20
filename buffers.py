from buffer_utils import (
    Experience,
    Experience_weight_idx,
    SumSegmentTree,
    MinSegmentTree,
)
from simple_config import ALPHA, BETA

from typing import Tuple, List

import numpy as np
import random


class Buffer(object):
    """Base Replay Buffer Object"""

    def __init__(self, size: int = 100000) -> None:
        """
        Create Replay buffer.
        Parameters
        ----------
        -   size : Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        """
        Returns number of elements in the buffer.
        """
        return len(self._storage)

    def __getitem__(self, keys: List[int]) -> List[Experience]:
        return [self._storage[key] for key in keys]

    def add(self, experience: Experience) -> None:
        """
        Adds a new (s, a, r, s', done) tuple to the buffer. This is a named tuple
        defined in buffer_utils.
        Parameters
        ----------
        -   experience : (state_t, action, reward, state_t+1, done)
        """

        if self._next_idx >= len(self._storage):
            self._storage.append(experience)
        else:
            self._storage[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences.
        Parameters
        ----------
        -   batch_size : How many transitions to sample.
        """
        idxes = np.random.choice(len(self), batch_size)
        sampled_experiences = [self._storage[idx] for idx in idxes]

        return sampled_experiences


class PrioritizedReplayBuffer(Buffer):
    """PER Buffer for Prioritized Experience Replay"""

    def __init__(
        self,
        size: int = 100000,
        alpha: float = ALPHA,
    ) -> None:
        """
        Create Prioritized Replay buffer.
        Parameters
        ----------
        -   size : Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        -   alpha: how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, experience: Experience_weight_idx) -> None:
        """
        Adds a new (s, a, r, s', done, weight, id) tuple to the buffer. This is
        a named tuple defined in buffer_utils.
        Parameters
        ----------
        -   experience : (state_t, action, reward, state_t+1, done)
        """
        idx = self._next_idx
        super().add(experience=experience)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha

    def _sample_proportional(self, batch_size: int) -> List[int]:
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(
        self, batch_size: int, beta: float = BETA
    ) -> List[Experience_weight_idx]:
        """
        Sample a batch of experiences. compared to ReplayBuffer.sample it also
        returns importance weights and idxes of sampled experiences.
        Parameters
        ----------
        -   batch_size : How many transitions to sample.
        -   beta : To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)

            """ Luca -------------------------"""
            self._storage[idx] = self._storage[idx]._replace(idx=idx)
            self._storage[idx] = self._storage[idx]._replace(weight=weight)
            """ ------------------------------ """

        weights = np.array(weights)

        sampled_experiences = [self._storage[idx] for idx in idxes]

        return sampled_experiences

    def update_priorities(self, idxes: List[int], priorities: List[float]) -> None:
        """
        Update priorities of sampled transitions. sets priority of transition
        at index idxes[i] in buffer to priorities[i].
        Parameters
        ----------
        -   idxes : List of idxes of sampled transitions
        -   priorities : List of updated priorities corresponding to
            transitions at the sampled idxes denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha

            self._max_priority = max(self._max_priority, priority)
