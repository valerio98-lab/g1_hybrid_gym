# g1_hybrid_gym/wrappers/task_env_wrapper.py
from __future__ import annotations

import torch
import numpy as np
from typing import Optional

# IVecEnv è l'interfaccia che rl_games si aspetta
from rl_games.common.vecenv import IVecEnv

import gym 
import gymnasium

class TaskEnvWrapper(IVecEnv):
    """
    Wrapper per il Task Learning compatibile con RL-Games e IsaacLab.
    Si posiziona SOPRA RlGamesVecEnvWrapper.
    """
    def __init__(
        self,
        env, # Questo è l'ambiente avvolto da RlGamesVecEnvWrapper
        a2c_network,
        s_dim: int,
        num_active_codebooks: int,
        codebook_size: int,
    ):
        self.env = env
        self.a2c_network = a2c_network
        self._s_dim = s_dim
        self._num_active_codebooks = num_active_codebooks
        self._codebook_size = codebook_size

        # Buffer per l'ultima osservazione raw (necessaria per il decoder)
        self._last_obs_raw: Optional[torch.Tensor] = None

        self.action_space = gym.spaces.MultiDiscrete(
            [codebook_size] * num_active_codebooks
        )

        self.observation_space = env.observation_space

    def get_env_info(self):
        # rl_games DiscreteA2CBase non supporta MultiDiscrete,
        # ma supporta Tuple di Discrete
        action_space = gym.spaces.Tuple(
            tuple(gym.spaces.Discrete(self._codebook_size) 
                for _ in range(self._num_active_codebooks))
        )
        return {
            "observation_space": self.observation_space,
            "action_space": action_space,
            "state_space": self.env.state_space if hasattr(self.env, 'state_space') else None,
            "agents": 1,
            "value_size": 1,
        }

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def reset(self):
        # RlGamesVecEnvWrapper restituisce un dict: {"obs": tensor, "states": tensor}
        res = self.env.reset()
        if isinstance(res, dict):
            self._last_obs_raw = res["obs"]
        else:
            self._last_obs_raw = res
        return res

    def step(self, action):
        """
        Riceve indici discreti (N, num_active) da rl_games e li converte in azioni fisiche.
        """
        if isinstance(action, torch.Tensor):
            indices = action.long()
        else:
            indices = torch.as_tensor(action, dtype=torch.long, device=self.device)

        if indices.dim() == 1 and self._num_active_codebooks == 1:
            indices = indices.unsqueeze(-1)

        # Estraiamo s_cur dall'osservazione raw per alimentare il decoder
        s_cur = self._last_obs_raw[..., : self._s_dim]

        # Pipeline congelata: Indici -> Azione Fisica (Joint Targets)
        with torch.no_grad():
            physical_action = self.a2c_network.indices_to_physical_action(s_cur, indices)

        # Passiamo l'azione fisica al wrapper IsaacLab sottostante
        # Ritorna: (obs_dict, reward, done, extras)
        obs_dict, reward, done, extras = self.env.step(physical_action)

        # Aggiorniamo la cache delle osservazioni per il prossimo step
        self._last_obs_raw = obs_dict["obs"]

        return obs_dict, reward, done, extras

    def close(self):
        return self.env.close()