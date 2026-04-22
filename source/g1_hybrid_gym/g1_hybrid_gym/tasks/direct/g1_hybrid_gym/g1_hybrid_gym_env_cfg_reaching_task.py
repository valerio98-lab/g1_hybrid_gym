"""
Environment config for the combined locomotion + arm reaching task.

Observation: s_cur(81) + g_task(8) = 89
  g_task = [vx, vy, dx_R, dy_R, dz_R, dx_L, dy_L, dz_L]
  where (dx_i, dy_i, dz_i) = target_i_body_frame - current_ee_i_body_frame

Action: 29 joint targets (physical actions from frozen decoder)
"""
from isaaclab.utils import configclass
from .g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg


@configclass
class G1HybridGymEnvReachingCfg(G1HybridGymEnvCfg):
    episode_length_s = 6.0

    # s_cur(81) + [vx, vy, dx_R, dy_R, dz_R, dx_L, dy_L, dz_L](8) = 89
    observation_space = 89

    # Physical joint targets (decoder output)
    action_space = 29

    # Base tracking rewards disabled — reaching env has its own reward
    rew_w_pose = 0.0
    rew_w_vel = 0.0
    rew_w_root_pos = 0.0
    rew_w_root_rot = 0.0
    rew_w_ee = 0.0
    rew_alive = 0.0

    min_height_reset = 0.5
