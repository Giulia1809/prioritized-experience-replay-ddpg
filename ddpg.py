from buffer_utils import Experience
from buffers import Buffer, PrioritizedReplayBuffer
from agent import Agent
from dataset import RLDataset

from typing import Tuple, OrderedDict, List

import gym
import numpy as np
import torch

from matplotlib import pyplot as plt


from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from simple_config import (
    ACTOR_LR,
    BATCH_SIZE,
    CRITIC_LR,
    ENV,
    EPISODE_LENGTH,
    GAMMA,
    SYNC_RATE,
    TAU,
    TRAIN_EPISODES,
    USE_PRIORITIZED_BUFFER,
    PRIORITIZED_REPLAY_EPS,
    TEST_EPISODES,
    RENDER,
)

torch.autograd.set_detect_anomaly(True)


class DDPGActor(LightningModule):
    """Actor Model of DDPG"""

    def __init__(self, state_dim: int, action_dim: int, action_max: float):
        """
        Args:
            -   state_dim : dimensions of the state space according to specific
                environment;
            -   action_dim : dimensions of the action space according to specific
                environment;
            -   action_max : upper limit of the actions values according to specific
                environment
        """
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        # TODO check if weights are actually initialized well like this
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-3e-3, 3e-3)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state: Tensor, action_max: float):
        return self.net(state) * action_max


class DDPGCritic(LightningModule):
    """Critic Model of DDPG"""

    def __init__(self, state_dim: int, action_dim: int):
        """
        Args:
            -   state_dim : dimensions of the state space according to specific
                environment;
            -   action_dim : dimensions of the action space according to specific
                environment;
        """
        super(DDPGCritic, self).__init__()

        self.state_l1 = nn.Linear(state_dim, 16)
        self.state_l2 = nn.Linear(16, 32)

        self.action_l1 = nn.Linear(action_dim, 32)

        self.act_st_l1 = nn.Linear(64, 256)
        self.act_st_l2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)

    def forward(self, state: Tensor, action):

        state_x = F.relu(self.state_l1(state))
        state_out = F.relu(self.state_l2(state_x))
        action_out = F.relu(self.action_l1(action))

        cat = torch.cat([state_out, action_out], dim=-1)

        act_st = F.relu(self.act_st_l1(cat))
        act_st = F.relu(self.act_st_l2(act_st))

        out = self.out(act_st)

        return out


# --------------------------------------------------------
# x = F.relu(self.l1(state))
# x = torch.cat([x, action], -1)
# x = F.relu(self.l2(x))
# x = self.l3(x)
# return x
# state_action = torch.cat([state, action], -1)
# # print(f"state_action shape: {state_action.shape}")
# x = F.relu(self.l1(state_action))
# x = F.relu(self.l2(x))
# x = self.l3(x)
# return x


class DDPG(LightningModule):
    """DDPG Model"""

    def __init__(
        self,
        logger: WandbLogger,
        batch_size: int = BATCH_SIZE,
        actor_lr: float = ACTOR_LR,
        critic_lr: float = CRITIC_LR,
        env: str = ENV,
        gamma: float = GAMMA,
        sync_rate: int = SYNC_RATE,
        episode_length: int = EPISODE_LENGTH,
        tau: float = TAU,
        train_episodes: float = TRAIN_EPISODES,
        use_prioritized_buffer: bool = USE_PRIORITIZED_BUFFER,
    ) -> None:
        """
        Args:
            -   batch_size : dimensions of the batches to use;
            -   actor_lr : learning rate of the actor model;
            -   critic_lr : learning rate of the critic model;
            -   env : Gym environment to use. Default: Pendulum-1;
            -   gamma : discount factor;
            -   tau : update the weights of the target networks with the weights
                of the original networks W_n * tau;
            -   train_episodes : on how many game episode training the model;
            -   use_prioritized_buffer : whether to use the Prioritized Experience
                Buffer or just a normal one;
        """
        super(DDPG, self).__init__()
        self.save_hyperparameters()

        """ Env and game parameters """
        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        self.action_upper_bound = self.env.action_space.high[0]

        """ 
        Initialize actor, critic, target actor and target critic. Targets are 
        assigned the same weights of the original networks. 
        """
        # Actor
        self.actor = DDPGActor(
            state_dim=obs_size, action_dim=n_actions, action_max=self.action_upper_bound
        )
        # Critic
        self.critic = DDPGCritic(state_dim=obs_size, action_dim=n_actions)
        # Target actor and weights equalization
        self.target_actor = DDPGActor(
            state_dim=obs_size, action_dim=n_actions, action_max=self.action_upper_bound
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        # Target critic and weights equalization
        self.target_critic = DDPGCritic(state_dim=obs_size, action_dim=n_actions)
        self.target_critic.load_state_dict(self.critic.state_dict())

        """
        Initialize buffer. If use_prioritized_buffer = 0 the normal buffer will 
        be used, otherwise the prioritized experience replay buffer.
        """
        self.buffer = (
            PrioritizedReplayBuffer()
            if self.hparams.use_prioritized_buffer
            else Buffer()
        )
        if self.hparams.use_prioritized_buffer:
            self.buffer = PrioritizedReplayBuffer()
        else:
            self.buffer = Buffer()

        self.test_step = 0

        # self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially
        fill up the replay buffer with experiences. Called by Lightning Callback
        just before the traing starts.

        Args:
            steps : number of random steps to populate the buffer with
        """
        dummy_agent = Agent(self.env, self.buffer)
        for _ in range(steps):
            dummy_agent.play_step(self.actor)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each
        action as an output.

        Args:
            x: environment state
        """
        out = self.actor(x)
        # out = self.actor(x.clone().detach().to(self.device))
        return out

    def _tensorify_gpufy(self, tensor_batch: List[Tensor]) -> Tensor:
        tensorified_list = tensor_batch[0].unsqueeze(dim=0)
        for i in range(1, len(tensor_batch)):
            tensorified_list = torch.cat(
                (
                    tensorified_list,
                    tensor_batch[i].unsqueeze(dim=0),
                ),
                0,
            )
        return tensorified_list

    def ddpg_loss(
        self, batch: List[Experience], module: "str"
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculates the mse loss using a mini batch from the replay buffer. See
        pseudo-code formulas. #TODO devo sistemare sta funzione che è uno spaghetti code di merda

        Args:
            batch: current mini batch of replay data
        """
        if module == "critic":
            (
                state_batch,
                action_batch,
                reward_batch,
                dones,
                next_state_batch,
                weights,
                idxes,
            ) = ([] for i in range(7))
            for exp in batch:
                state_batch.append(exp.state.float())
                action_batch.append(exp.action.float())
                reward_batch.append(float(exp.reward))
                dones.append(exp.done)
                next_state_batch.append(exp.new_state.float())

                if USE_PRIORITIZED_BUFFER:
                    idxes.append(exp.idx)
                    weights.append(exp.weight.float())

            if USE_PRIORITIZED_BUFFER:
                """They do not use importance sampling weights, therefore
                they set all importance sampling weights to 1"""
                weights *= 0 + 1
                weights_tensor = self._tensorify_gpufy(weights)
                weights_tensor = torch.sqrt(weights_tensor)

            next_state_batch_tensor = self._tensorify_gpufy(next_state_batch)
            state_batch_tensor = self._tensorify_gpufy(state_batch)
            action_batch_tensor = self._tensorify_gpufy(action_batch)

            target_actions = self.target_actor(
                next_state_batch_tensor, self.action_upper_bound
            )

            reward_batch = (
                torch.tensor(reward_batch, device=self.device)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
            )  # FIXME double unsueeze was single

            target_critic_value = self.target_critic(
                next_state_batch_tensor, target_actions
            )  # FIXME .squeeze(dim=1)

            y = reward_batch + self.hparams.gamma * target_critic_value

            critic_value = self.critic(
                state_batch_tensor, action_batch_tensor
            )  # FIXME .squeeze(dim=-1)

            td_errors = y - critic_value

            # Create a zero tensor
            zero_tensor = torch.zeros(td_errors.shape, device=self.device)
            if USE_PRIORITIZED_BUFFER:
                # Weight TD errors
                weighted_TD_errors = torch.mul(td_errors, weights_tensor)
                # Compute critic loss, MSE of weighted TD_r
                critic_loss = F.mse_loss(weighted_TD_errors, zero_tensor)
            else:
                critic_loss = F.mse_loss(td_errors, zero_tensor)

            if USE_PRIORITIZED_BUFFER:
                td_errors = td_errors.detach().cpu()
                new_priorities = (
                    torch.mean(torch.abs(td_errors), -1, keepdim=False)
                    + PRIORITIZED_REPLAY_EPS
                ).squeeze(-1)
                # exit()

                # new_priorities = np.abs(td_errors) + PRIORITIZED_REPLAY_EPS
                self.buffer.update_priorities(idxes, new_priorities.tolist())

            return critic_loss

        elif module == "actor":
            state_batch, action_batch, reward_batch, dones, next_state_batch = (
                [] for i in range(5)
            )
            for exp in batch:
                state_batch.append(exp.state)
                action_batch.append(exp.action)
                reward_batch.append(float(exp.reward))
                dones.append(exp.done)
                next_state_batch.append(exp.new_state)

            state_batch_tensor = self._tensorify_gpufy(state_batch)

            actions = self.actor(state_batch_tensor, self.action_upper_bound)
            critic_value = self.critic(
                state_batch_tensor, actions
            )  # FIXME .squeeze(dim=-1)

            actor_loss = -torch.mean(critic_value)

            return actor_loss

    def _update_target(
        self, target_net: LightningModule, source_net: LightningModule, tau: float
    ) -> None:
        """
        Soft updates the weights of target net with source net.
        Arguments
        ----------
            -   target_net : target networks to which infer the source net's weights
            -   source_net : target networks from which infer the weights to target
            -   tau : weighting factor for the weights to be inferred

        ****** Super hacky thing but apparently no other solutions exists ******
        """
        # Find all the layers containing the weights avoiding the bias layers
        # for layer_name, layer_content in source_net.state_dict().items():
        #     if "weight" in layer_name:
        #         # target_net.state_dict()[layer_name] *= (1 - tau) + (tau * layer_content)
        #         target_net.state_dict()[layer_name] = target_net.state_dict()[
        #             layer_name
        #         ] * (1 - tau) + (tau * layer_content)

        for q_param, target_param in zip(
            source_net.parameters(), target_net.parameters()
        ):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * q_param)

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor],
        nb_batch,
        optimizer_idx,
    ) -> Tuple[OrderedDict, OrderedDict]:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved.

        Args:
            batch : current mini batch of replay data
            nb_batch : batch number
        """
        # Actor optimizer idx
        if optimizer_idx == 0:

            # calculates training loss
            actor_loss = self.ddpg_loss(batch, "actor")

            # Soft update
            if self.global_step % self.hparams.sync_rate == 0:
                self._update_target(self.target_actor, self.actor, self.hparams.tau)

            self.log_dict(
                {
                    "train_actor_loss": actor_loss,
                }
            )
            self.log("steps", self.global_step, logger=False, prog_bar=True)

            return actor_loss

        # Critic optimizer idx
        elif optimizer_idx == 1:
            # calculates training loss
            critic_loss = self.ddpg_loss(batch, "critic")

            # FIXME Soft update
            if self.global_step % self.hparams.sync_rate == 0:
                self._update_target(self.target_critic, self.critic, self.hparams.tau)
            self.log_dict(
                {
                    "train_critic_loss": critic_loss,
                }
            )
            return critic_loss

        else:
            raise ValueError("Optimizers_idx can only be 0 or 1")

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.hparams.critic_lr,
            weight_decay=1e-2,  # FIXME check il wd is ok to use
        )
        actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.hparams.actor_lr
        )

        return [actor_optimizer, critic_optimizer]

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(
            self.buffer,
            self.hparams.batch_size,
            self.env,
            self.actor,
            TRAIN_EPISODES,
            self.hparams.logger,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def _test_loop(self) -> None:
        env = self.env
        device = self.actor.device

        rewards = []
        for episode_n in range(TEST_EPISODES):

            state = env.reset()
            state = torch.from_numpy(state).clone().detach().to(device)

            episode_reward = 0
            done = 0
            while not done:

                if RENDER:
                    env.render()

                    # img = env.render(mode="rgb_array")
                    # plt.imshow(img)
                    # plt.savefig("playing/roll_" + str(self.test_step) + ".png")
                    # self.test_step += 1

                action_upper_bound = env.action_space.high[0]
                action_lower_bound = env.action_space.low[0]

                sampled_action = self.actor(state, action_upper_bound)
                legal_action = np.clip(
                    sampled_action.detach().cpu().numpy(),
                    action_lower_bound,
                    action_upper_bound,
                )
                legal_action = torch.tensor(legal_action).to(device)

                new_state, reward, done, _ = env.step(
                    legal_action.detach().cpu().numpy()
                )
                new_state = torch.from_numpy(new_state).to(device)
                state = new_state

                episode_reward += reward
            rewards.append(episode_reward)

            print(f"Episode {episode_n} reward: {episode_reward}")

        print(f"Mean reward over {TEST_EPISODES} episodes: {np.mean(rewards)}")

    def test(self) -> None:
        self._test_loop()
