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
        num_envs=8192, env_spacing=4.0, replicate_physics=True
    )

    # - action scale
    # action_scale = 0.5  # [N]

    # - reward scales (Coefficienti per l'esponenziale: exp(-w * err))
    # Valori ispirati a DeepMimic / Paper Appendix B.2
    rew_w_pose = 2.0  # Paper Eq.10: r_q uses -2
    rew_w_vel = 0.5  # Paper Eq.10: r_alpha uses -0.5 (Abbassato per stabilità infatti usano 0.1 nel codice)
    rew_w_root_pos = 10.0  # Paper Eq.10: r_root uses -10.
    rew_w_root_rot = 0  # Non nel paper ma utile per stabilità. Al momento non usiamo rotazione del root
    rew_w_ee = 40.0  # Paper Eq.10: r_ee uses -40 (Molto alto!)

    rew_alive = 0.0  # Additivo. Attualmente non usato, spingeva il robot a stare in piedi ignorando il tracking

    # - reset states/conditions
    min_height_reset = 0.5  # reset if robot falls below this height [m]
