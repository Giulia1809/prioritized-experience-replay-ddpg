from sys import maxsize
from buffers import Buffer
from agent_utils import OUActionNoise
from buffer_utils import Experience, Experience_weight_idx
from simple_config import OU_NOISE_STD, RENDER, USE_PRIORITIZED_BUFFER

import gym
import numpy as np
from typing import Tuple

import torch
from torch import nn


class Agent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(self, env: gym.Env, buffer: Buffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.buffer = buffer
        self.reset()
        self.state = self.env.reset()
        self.state = torch.from_numpy(self.state)

        _std = OU_NOISE_STD
        self.noise_model = OUActionNoise(
            mean=np.zeros(1) * 1.2, std_deviation=float(_std) * np.ones(1)
        )

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset()
        self.state = torch.from_numpy(self.state)

    def get_action(self, net: nn.Module, device: str) -> int:
        """
        Using the given network, decide what action to carry out using the
        ornstein_uhlenbeck noise to perturb the action.

        Args:
            -   net: actor DDPG network
            -   epsilon: value to determine likelihood of taking a random action
        """
        action_upper_bound = self.env.action_space.high[0]
        action_lower_bound = self.env.action_space.low[0]

        state = self.state.clone().detach().to(device)
        sampled_actions = net(state, action_upper_bound) + torch.tensor(
            self.noise_model()
        ).to(device)

        # make sure action is within bounds
        legal_action = np.clip(
            sampled_actions.cpu().numpy(),
            action_lower_bound,
            action_upper_bound,
        )

        return torch.tensor(legal_action).to(device)

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
    ) -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment.

        Args:
            -   net: Actor DDPG network
        """
        if RENDER:
            self.env.render()
        action = self.get_action(net, device=net.device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action.cpu().numpy())
        new_state = torch.from_numpy(new_state).to(net.device)

        self.state = self.state.to(net.device)

        if USE_PRIORITIZED_BUFFER:
            exp = Experience_weight_idx(
                self.state.cpu(),
                action.cpu(),
                reward,
                done,
                new_state.cpu(),
                1 / self.buffer._maxsize,
                -1,
            )
        else:
            exp = Experience(
                self.state.cpu(),
                action.cpu(),
                reward,
                done,
                new_state.cpu(),
            )

        self.buffer.add(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done
