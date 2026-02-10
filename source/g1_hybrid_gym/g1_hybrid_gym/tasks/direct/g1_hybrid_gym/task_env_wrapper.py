# g1_hybrid_gym/wrappers/task_env_wrapper.py
"""
Environment wrapper for task learning.

Sits between rl_games and G1HybridGymEnvTask:
  - rl_games agent outputs: codebook indices  (N, num_active_codebooks)
  - IsaacLab env expects:   joint targets     (N, 29)

This wrapper intercepts the discrete indices, uses the frozen
prior + codebook + decoder (via TaskA2CNetwork.indices_to_physical_action)
to produce physical actions, and forwards those to the underlying env.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch


class TaskEnvWrapper(gym.Wrapper):
    """
    Args:
        env:            G1HybridGymEnvTask instance
        a2c_network:    TaskA2CNetwork (from wrapper_task_ppo.py)
        s_dim:          Dimension of proprioceptive state in obs
        num_active_codebooks: How many codebooks the high-level uses
        codebook_size:  Number of entries per codebook
    """

    def __init__(
        self,
        env,
        a2c_network,
        s_dim: int,
        num_active_codebooks: int,
        codebook_size: int,
    ):
        super().__init__(env)
        self.a2c_network = a2c_network
        self._s_dim = s_dim
        self._num_active_codebooks = num_active_codebooks
        self._codebook_size = codebook_size

        # Cache last obs to extract s_cur at step time
        self._last_obs_policy: Optional[torch.Tensor] = None

        # Override action space so rl_games sees MultiDiscrete
        self.action_space = gym.spaces.MultiDiscrete(
            [codebook_size] * num_active_codebooks
        )

        # Observation space unchanged
        obs_dim = env.unwrapped.cfg.observation_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}
        if isinstance(obs, dict):
            self._last_obs_policy = obs["policy"]
        else:
            self._last_obs_policy = obs
        return self._last_obs_policy

    def step(self, action):
        """
        action: (N, num_active_codebooks) — discrete indices from rl_games
        """
        if isinstance(action, torch.Tensor):
            indices = action.long()
        else:
            indices = torch.as_tensor(action, dtype=torch.long, device=self._get_device())

        if indices.dim() == 1 and self._num_active_codebooks == 1:
            indices = indices.unsqueeze(-1)

        # Extract s_cur from cached obs
        s_cur = self._last_obs_policy[..., : self._s_dim]

        # Frozen pipeline: indices → physical action
        with torch.no_grad():
            physical_action = self.a2c_network.indices_to_physical_action(s_cur, indices)

        # Step the real physics env
        obs, reward, terminated, truncated, extras = self.env.step(physical_action)

        if isinstance(obs, dict):
            self._last_obs_policy = obs["policy"]
        else:
            self._last_obs_policy = obs
        
        done = (terminated | truncated).to(dtype=torch.float32)

        return self._last_obs_policy, reward, done, {} if extras is None else extras

    def _get_device(self):
        if hasattr(self.env, "device"):
            return self.env.device
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "device"):
            return self.env.unwrapped.device
        return "cuda:0"