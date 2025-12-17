# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class G1HybridGymEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 5.0
    # - spaces definition
    action_space = 29
    observation_space = 138
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, gravity=(0.0, 0.0, -9.81)
    )

    # robot(s)
    robot_cfg: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=20, env_spacing=4.0, replicate_physics=True
    )

    # - action scale
    action_scale = 1.0  # [N]
    # - reward scales
    rew_w_pose = 1.0
    rew_w_vel = 0.01
    rew_alive = 0.1
    # - reset states/conditions
    min_height_reset = 0.5  # reset if robot falls below this height [m]
