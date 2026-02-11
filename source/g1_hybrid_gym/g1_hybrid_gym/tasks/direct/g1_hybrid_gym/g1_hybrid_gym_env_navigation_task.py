# g1_hybrid_gym/tasks/direct/g1_hybrid_gym/g1_hybrid_gym_env_task.py
"""
Velocity-command tracking environment for task learning.

Goal observation: g = (vx_cmd, vy_cmd, ω_cmd) in body frame
Reward: tracking linear + angular velocity (inspired by paper Eq.16)
Early termination: only on fall (no reference trajectory)
Command resampling: every N seconds within episode for transition learning
"""
from __future__ import annotations

import torch
from typing import Dict, Optional

from .g1_hybrid_gym_env_base import G1HybridGymEnvBase
from .g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
from g1_hybrid_prior.helpers import (
    quat_normalize,
    quat_rotate_inv,
    quat_rotate,
    quat_mul
)
from isaaclab.envs import DirectRLEnv



class G1HybridGymEnvTask(G1HybridGymEnvBase):
    """
    Velocity-command tracking environment for task learning phase.

    Unlike PPO/Imitation envs that track reference motion frames,
    this env receives velocity commands and the agent must discover
    appropriate motion skills from the latent space.

    Observation: [s_cur, g_task]
      - s_cur: same proprioceptive state as base (h, quat, vel, angvel, joints, joint_vel)
      - g_task: (vx_cmd, vy_cmd, omega_cmd) in body frame — 3D

    Reward: velocity tracking (no reference motion needed)
    """

    def __init__(self, cfg: G1HybridGymEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        params = self._read_dataset_params()
        task_cfg = params.get("task_learning", {})

        self.vx_range = tuple(task_cfg.get("vx_range", [0.0, 1.5]))
        self.vy_range = tuple(task_cfg.get("vy_range", [-0.3, 0.3]))
        self.omega_range = tuple(task_cfg.get("omega_range", [-0.8, 0.8]))

        # Command resampling interval (seconds)
        self.cmd_resample_seconds = float(task_cfg.get("cmd_resample_seconds", 2.5))
        self.cmd_resample_steps = int(
            self.cmd_resample_seconds / (self.cfg.sim.dt * self.cfg.decimation)
        )

        self.rew_w_lin_vel = float(task_cfg.get("rew_w_lin_vel", 1.0))
        self.rew_w_ang_vel = float(task_cfg.get("rew_w_ang_vel", 0.5))
        self.rew_w_lin_vel_penalty = float(task_cfg.get("rew_w_lin_vel_penalty", 2.0))
        self.rew_w_ang_vel_penalty = float(task_cfg.get("rew_w_ang_vel_penalty", 1.0))

        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.cmd_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        J = len(self.dataset.robot_cfg.joint_order)
        self.s_dim = 1 + 4 + 3 + 3 + J + J  # 69 for G1
        self.task_goal_dim = 3

        print(
            f"[{self.__class__.__name__}] Task env initialized. "
            f"s_dim={self.s_dim}, goal_dim={self.task_goal_dim}, "
            f"vx_range={self.vx_range}, vy_range={self.vy_range}, "
            f"omega_range={self.omega_range}, "
            f"cmd_resample_steps={self.cmd_resample_steps}"
        )

    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample velocity commands for given environments."""
        n = len(env_ids)
        vx = torch.empty(n, device=self.device).uniform_(*self.vx_range)
        vy = torch.empty(n, device=self.device).uniform_(*self.vy_range)
        omega = torch.empty(n, device=self.device).uniform_(*self.omega_range)
        self.vel_cmd[env_ids] = torch.stack([vx, vy, omega], dim=-1)
        self.cmd_timer[env_ids] = 0

    def _get_observations(self) -> dict:
        """
        Override base: goal = velocity command instead of reference state difference.
        Returns obs["policy"] = [s_cur, vel_cmd]
        """
        root_link_state = self.robot.data.root_link_state_w  # (N, 13)
        root_pos_w = root_link_state[:, 0:3]
        root_quat_wxyz = quat_normalize(root_link_state[:, 3:7])

        v_link_world = root_link_state[:, 7:10]
        w_link_world = root_link_state[:, 10:13]

        root_lin_vel_body = quat_rotate_inv(root_quat_wxyz, v_link_world)
        root_ang_vel_body = quat_rotate_inv(root_quat_wxyz, w_link_world)

        env_origins = self.scene.env_origins
        h = (root_pos_w - env_origins)[:, 2:3]

        joint_pos = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]
        joint_vel = self.robot.data.joint_vel[:, self.dataset_to_isaac_indexes]

        s_cur = torch.cat(
            (h, root_quat_wxyz, root_lin_vel_body, root_ang_vel_body, joint_pos, joint_vel),
            dim=-1,
        )

        g_task = self.vel_cmd  # (N, 3)
        obs = torch.cat((s_cur, g_task), dim=-1)

        # Cache for reward computation
        self._cached_lin_vel_body = root_lin_vel_body
        self._cached_ang_vel_body = root_ang_vel_body

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Velocity tracking reward:
          r = w_lin * exp(-r_lin * ||v_xy - v_cmd_xy||²)
            + w_ang * exp(-r_ang * (ω_z - ω_cmd)²)
        """
        v_body = self._cached_lin_vel_body
        w_body = self._cached_ang_vel_body

        vx_cur, vy_cur = v_body[:, 0], v_body[:, 1]
        wz_cur = w_body[:, 2]

        vx_cmd, vy_cmd, omega_cmd = self.vel_cmd[:, 0], self.vel_cmd[:, 1], self.vel_cmd[:, 2]

        lin_vel_err = (vx_cur - vx_cmd) ** 2 + (vy_cur - vy_cmd) ** 2
        r_lin = torch.exp(-self.rew_w_lin_vel_penalty * lin_vel_err)

        ang_vel_err = (wz_cur - omega_cmd) ** 2
        r_ang = torch.exp(-self.rew_w_ang_vel_penalty * ang_vel_err)

        reward = self.rew_w_lin_vel * r_lin + self.rew_w_ang_vel * r_ang

        # Command resampling
        self.cmd_timer += 1
        resample_mask = self.cmd_timer >= self.cmd_resample_steps
        if resample_mask.any():
            resample_ids = resample_mask.nonzero(as_tuple=False).squeeze(-1)
            self._resample_commands(resample_ids)

        # Debug buffers
        with torch.no_grad():
            self._dbg_lin_vel_err = lin_vel_err
            self._dbg_ang_vel_err = ang_vel_err

        return reward

    def _get_dones(self):
        """
        Task learning: only fall detection, no reference trajectory termination.
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        root_z_rel = (
            self.robot.data.root_link_state_w[:, 2] - self.scene.env_origins[:, 2]
        )
        fallen = root_z_rel < self.cfg.min_height_reset
        terminated = fallen

        self._dbg_fallen = fallen
        self._cached_ref_tensors = None

        return terminated, time_out


    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        DirectRLEnv._reset_idx(self, env_ids)

        n = len(env_ids)

        # RSI
        min_episode_frames = 60
        max_start = max(0, self.max_frame_idx - min_episode_frames)
        random_starts = torch.randint(0, max_start + 1, (n,), device=self.device)
        self.ref_frame_idx[env_ids] = random_starts

        root_pos_0 = self._ref_root_pos[random_starts]
        root_quat_0 = self._ref_root_quat_wxyz[random_starts]
        root_lin_vel_0_body = self._ref_root_lin_vel[random_starts].clone()
        root_ang_vel_0_body = self._ref_root_ang_vel[random_starts].clone()
        joints_0 = self._ref_joints[random_starts]
        joint_vel_0 = self._ref_joint_vel[random_starts]

        # Heading Randomization 
        rand_yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * torch.pi
        cy = torch.cos(rand_yaw * 0.5)
        sy = torch.sin(rand_yaw * 0.5)
        zeros = torch.zeros_like(cy)
        q_rand_yaw = torch.stack([cy, zeros, zeros, sy], dim=-1)

        root_quat_w = quat_normalize(quat_mul(q_rand_yaw, root_quat_0))

        root_lin_vel_0_body += (torch.rand(n, 3, device=self.device) * 2 - 1) * 0.1
        ang_vel_noise = (torch.rand(n, 3, device=self.device) * 2 - 1) * 0.05
        ang_vel_noise[:, 2] *= 0.1
        root_ang_vel_0_body += ang_vel_noise

        env_origins = self.scene.env_origins[env_ids]
        root_pos_w = root_pos_0 + env_origins

        v_link_w = quat_rotate(root_quat_w, root_lin_vel_0_body)
        w_w = quat_rotate(root_quat_w, root_ang_vel_0_body)
        r_body = self.robot.data.body_com_pos_b[env_ids, 0]
        r_w = quat_rotate(root_quat_w, r_body)
        v_com_w = v_link_w + torch.linalg.cross(w_w, r_w, dim=-1)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] = root_pos_w
        default_root_state[:, 3:7] = root_quat_w
        default_root_state[:, 7:10] = v_com_w
        default_root_state[:, 10:13] = w_w

        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_joint_pos[:, self.dataset_to_isaac_indexes] = joints_0
        default_joint_vel[:, self.dataset_to_isaac_indexes] = joint_vel_0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)

        self._resample_commands(env_ids)
        self._cached_ref_tensors = None

    def step(self, action):
        """Override to add task-specific logging."""
        obs, rew, terminated, truncated, extras = super().step(action)

        done = terminated | truncated
        if done.any() and extras is not None:
            log = extras.setdefault("log", {})
            ids = done.nonzero(as_tuple=False).squeeze(-1)

            if hasattr(self, "_dbg_lin_vel_err") and self._dbg_lin_vel_err is not None:
                log["vel_lin_err_mean"] = self._dbg_lin_vel_err[ids].mean().item()
            if hasattr(self, "_dbg_ang_vel_err") and self._dbg_ang_vel_err is not None:
                log["vel_ang_err_mean"] = self._dbg_ang_vel_err[ids].mean().item()

        return obs, rew, terminated, truncated, extras