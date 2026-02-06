# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="ExpertPPO-G1HybridGymEnv-v0",
    entry_point=f"{__name__}.g1_hybrid_gym_env_ppo:G1HybridGymEnvPPO",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_hybrid_gym_env_cfg:G1HybridGymEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_expert_ppo_cfg.yaml",
    },
)

gym.register(
    id="TaskLearningPPO-G1HybridGymEnv-v0",
    entry_point=f"{__name__}.g1_hybrid_gym_env_ppo:???",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_hybrid_gym_env_cfg:G1HybridGymEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_task_learning_ppo_cfg.yaml",
    },
)
