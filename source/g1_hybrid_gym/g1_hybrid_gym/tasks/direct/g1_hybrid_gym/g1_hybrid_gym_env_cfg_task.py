# g1_hybrid_gym/tasks/direct/g1_hybrid_gym/g1_hybrid_gym_env_task_cfg.py
"""
Environment config for task learning.
Observation: s_cur(69) + vel_cmd(3) = 72
Action: 29 joint targets (physical actions from frozen decoder)
"""
from isaaclab.utils import configclass
from .g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg


@configclass
class G1HybridGymEnvTaskCfg(G1HybridGymEnvCfg):
    episode_length_s = 6.0

    # Observation: s_cur(69) + vel_cmd(3) = 72
    observation_space = 72

    # Action: physical joint targets (decoder output, NOT codebook indices)
    # The env wrapper handles the discrete→continuous translation
    action_space = 29

    # Base reward weights disabled (task env has its own reward)
    rew_w_pose = 0.0
    rew_w_vel = 0.0
    rew_w_root_pos = 0.0
    rew_w_root_rot = 0.0
    rew_w_ee = 0.0
    rew_alive = 0.0

    # Fall detection threshold
    min_height_reset = 0.5