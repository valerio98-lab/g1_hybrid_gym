# g1_hybrid_gym/wrappers/wrapper_task_ppo.py
"""
rl_games wrapper for task-learning PPO using ModelA2CMultiDiscrete.

Imports TaskLearningBlock and TaskCritic from g1_hybrid_prior,
adapts them to the interface that ModelA2CMultiDiscrete expects:

    logits, value, states = self.a2c_network(input_dict)

where logits is a list of (B, codebook_size) tensors (one per active codebook).

rl_games then handles Categorical sampling, log_prob, entropy, and PPO
clipping entirely on its own.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CMultiDiscrete

from g1_hybrid_prior.models.task_learning_block import (
    TaskLearningBlock,
    TaskCritic,
)


class TaskA2CNetwork(nn.Module):
    """
    Thin adapter that wraps TaskLearningBlock + TaskCritic
    into the (logits_list, value, states) interface that
    ModelA2CMultiDiscrete.Network.forward() expects.

    Also exposes indices_to_physical_action() for the env wrapper.
    """

    def __init__(self, task_block: TaskLearningBlock, critic: TaskCritic, s_dim: int, task_goal_dim: int):
        super().__init__()
        self.task_block = task_block
        self.critic = critic
        self.s_dim = s_dim
        self.task_goal_dim = task_goal_dim

    def forward(self, input_dict: dict):
        """
        Called by ModelA2CMultiDiscrete.Network.forward().
        obs is already normalized by BaseModelNetwork.norm_obs().

        Returns:
            logits:  list of (B, codebook_size) tensors
            value:   (B, 1)
            states:  None (no RNN)
        """
        obs = input_dict["obs"]  # (B, s_dim + task_goal_dim), already normalized
        s = obs[..., : self.s_dim]
        g = obs[..., self.s_dim :]

        # High-level policy logits
        hl_out = self.task_block.high_level(s, g)
        logits = hl_out["logits"]  # (B, num_active_codebooks, codebook_size)

        # ModelA2CMultiDiscrete expects a list of (B, codebook_size)
        logits_list = [logits[:, q, :] for q in range(logits.shape[1])]

        # Critic
        value = self.critic(s, g)  # (B, 1)

        return logits_list, value, None

    @torch.no_grad()
    def indices_to_physical_action(self, s: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert codebook indices to physical joint actions via frozen pipeline.
        Used by TaskEnvWrapper.

        Args:
            s:       (B, s_dim) proprioceptive state
            indices: (B, num_active_codebooks) long

        Returns:
            action:  (B, action_dim) joint position targets
        """
        zp = self.task_block.imitation.prior(s)
        y_bar = self.task_block._lookup_codebook(indices)
        z_bar = y_bar + zp
        action = self.task_block.imitation.decoder(s, z_bar)
        return action

    def is_rnn(self):
        return False

    def get_default_rnn_state(self):
        return None

    def get_aux_loss(self):
        return None

    def get_value_layer(self):
        return None


class TaskPolicyWrapper(ModelA2CMultiDiscrete):
    """
    rl_games model that builds a TaskA2CNetwork from config.

    Expected keys in config (forwarded from rl_games yaml):
        task_goal_dim:        int  (default 3)
        imitation_checkpoint: str  (path to imitation .pt)
        num_active_codebooks: int  (default 1, read from TaskLearning.yaml)
        physical_action_dim:  int  (default 29)
    """

    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        obs_shape = config["input_shape"]  # (72,)
        if len(obs_shape) != 1:
            raise RuntimeError(f"[TaskPolicyWrapper] Only flat obs, got {obs_shape}")

        full_obs_dim = obs_shape[0]
        task_goal_dim = int(config.get("task_goal_dim", 3))
        s_dim = full_obs_dim - task_goal_dim
        physical_action_dim = int(config.get("physical_action_dim", 29))
        imitation_ckpt = config["imitation_checkpoint"]

        device = config.get("device", "cuda:0")

        task_block = TaskLearningBlock(
            s_dim=s_dim,
            goal_dim=s_dim,  # imitation used goal_dim = s_dim
            task_goal_dim=task_goal_dim,
            action_dim=physical_action_dim,
            imitation_ckpt_path=imitation_ckpt,
        ).to(device)

        critic = TaskCritic(
            s_dim=s_dim,
            goal_dim=task_goal_dim,
        ).to(device)

        a2c_network = TaskA2CNetwork(
            task_block=task_block,
            critic=critic,
            s_dim=s_dim,
            task_goal_dim=task_goal_dim,
        )

        codebook_size = task_block.codebook_size
        num_active = task_block.num_active_codebooks
        print(
            f"[TaskPolicyWrapper] s_dim={s_dim}, task_goal_dim={task_goal_dim}, "
            f"action_dim={physical_action_dim}, "
            f"codebook_size={codebook_size}, num_active_codebooks={num_active}"
        )

        value_size = config.get("value_size", 1)
        normalize_value = config["normalize_value"]
        normalize_input = config["normalize_input"]

        return self.Network(
            a2c_network,
            obs_shape=obs_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            value_size=value_size,
        )