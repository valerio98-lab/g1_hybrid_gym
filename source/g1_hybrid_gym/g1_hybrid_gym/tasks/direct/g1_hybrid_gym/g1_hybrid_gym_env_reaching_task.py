"""
Combined locomotion + arm reaching task.

The robot must simultaneously:
  1. Track a commanded linear velocity (vx, vy) in body frame
  2. Move both arm end-effectors to target positions sampled in body frame

Task goal: [vx, vy, dx_R, dy_R, dz_R, dx_L, dy_L, dz_L]  (8D)
  where (dx_i, dy_i, dz_i) = target_i - current_ee_i  in body frame

Scenarios (controlled via fixed_* attributes or --phases in play):
  1. fixed_vx=0, fixed_vy=0 → robot stands still and reaches with arms
  2. fixed_vx>0             → robot walks while moving arms to targets
  3. fixed_vx>0, fixed_ee_targets set to wide positions → arms-as-wings while walking
"""
from __future__ import annotations

from typing import Optional

import torch
import einops

from isaaclab.envs import DirectRLEnv

from .g1_hybrid_gym_env_base import G1HybridGymEnvBase
from g1_hybrid_prior.helpers import (
    quat_normalize,
    quat_rotate,
    quat_rotate_inv,
    quat_mul,
)

# arm EE names — must match robots.yaml ee_link_names
_DEFAULT_ARM_EE_NAMES = ["right_hand_palm_link", "left_hand_palm_link"]


class G1HybridGymEnvReaching(G1HybridGymEnvBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        params = self._read_dataset_params()

        # ── velocity command ranges ────────────────────────────────────────
        self.vx_range: list[float] = params.get("vx_range", [-0.5, 1.0])
        self.vy_range: list[float] = params.get("vy_range", [-0.3, 0.3])

        # ── EE target workspace in body frame (metres) ────────────────────
        ee_cfg = params.get("reaching_ee_cfg", {})
        self.ee_x_range: list[float] = ee_cfg.get("x_range", [-0.1, 0.4])
        self.ee_z_range: list[float] = ee_cfg.get("z_range", [0.6, 1.4])
        # y convention: body frame +y = robot's left, -y = robot's right
        self._ee_y_range_by_side = {
            "right": ee_cfg.get("y_range_right", [-0.7, -0.25]),
            "left":  ee_cfg.get("y_range_left",  [ 0.25,  0.7]),
        }

        # ── command resample period ────────────────────────────────────────
        resample_s = float(params.get(
            "cmd_resample_seconds_reaching",
            params.get("cmd_resample_seconds", 3.0),
        ))
        control_dt = self.cfg.sim.dt * self.cfg.decimation
        self.cmd_resample_steps: int = max(1, int(resample_s / control_dt))

        # ── arm EE indices ─────────────────────────────────────────────────
        arm_ee_names = params.get("reaching_arm_ee_names", _DEFAULT_ARM_EE_NAMES)
        all_body_names = self.robot.body_names
        arm_ee_isaac: list[int] = []
        for name in arm_ee_names:
            if name not in all_body_names:
                raise ValueError(
                    f"[{self.__class__.__name__}] Arm EE '{name}' not found "
                    f"in articulation. Available: {all_body_names}"
                )
            arm_ee_isaac.append(all_body_names.index(name))

        self.arm_ee_names: list[str] = list(arm_ee_names)
        self.arm_ee_isaac_indices = torch.tensor(
            arm_ee_isaac, device=self.device, dtype=torch.long
        )
        self.num_arm_ee: int = len(arm_ee_names)

        # Map each arm EE to its y-range (right vs left from name)
        self._arm_y_ranges: list[list[float]] = []
        for name in arm_ee_names:
            nl = name.lower()
            side = "right" if ("right" in nl) else "left"
            self._arm_y_ranges.append(self._ee_y_range_by_side[side])

        # ── task goal dimension ────────────────────────────────────────────
        # [vx, vy] + num_arm_ee * [dx, dy, dz]
        self.task_goal_dim: int = 2 + self.num_arm_ee * 3
        self.s_dim: int = self.CUR_OBS_DIM  # convenience alias

        # ── command buffers ────────────────────────────────────────────────
        self.vel_cmd = torch.zeros(self.num_envs, 2, device=self.device)
        # EE targets in body frame: (N, num_arm_ee, 3)
        self.ee_target_body = torch.zeros(
            self.num_envs, self.num_arm_ee, 3, device=self.device
        )
        self.cmd_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ── reward weights ─────────────────────────────────────────────────
        self.rew_w_lin_vel      = float(params.get("rew_w_lin_vel_reaching",     0.3))
        self.rew_lin_vel_pen    = float(params.get("rew_lin_vel_penalty_reaching", 1.0))
        self.rew_w_ee_reach     = float(params.get("rew_w_ee_reaching",           0.7))
        self.rew_ee_pen         = float(params.get("rew_ee_penalty_reaching",      5.0))

        # ── play overrides (set externally before running) ─────────────────
        # Locomotion
        self.fixed_vx: Optional[float] = None
        self.fixed_vy: Optional[float] = None
        # EE targets: list of (x, y, z) per arm EE, or None to resample freely
        # Example wings pose: [(-0.05, -0.65, 1.0), (-0.05, 0.65, 1.0)]
        self.fixed_ee_targets: Optional[list[tuple[float, float, float]]] = None

        # ── debug buffers ──────────────────────────────────────────────────
        self._cached_lin_vel_body: Optional[torch.Tensor] = None
        self._cached_ee_pos_b: Optional[torch.Tensor] = None
        self._dbg_lin_vel_err: Optional[torch.Tensor] = None
        self._dbg_ee_dist: Optional[torch.Tensor] = None

        # Initial command sample
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._resample_commands(all_ids)

        print(
            f"[{self.__class__.__name__}] initialized. "
            f"arm_ee={self.arm_ee_names}  task_goal_dim={self.task_goal_dim}  "
            f"cmd_resample_steps={self.cmd_resample_steps} (~{resample_s:.1f}s)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Command sampling
    # ------------------------------------------------------------------

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)

        # Velocity
        if self.fixed_vx is not None:
            vx = torch.full((n,), self.fixed_vx, device=self.device)
            vy = torch.full(
                (n,), self.fixed_vy if self.fixed_vy is not None else 0.0,
                device=self.device,
            )
        else:
            vx = torch.empty(n, device=self.device).uniform_(*self.vx_range)
            limit_x = torch.where(
                vx >= 0,
                torch.tensor(self.vx_range[1], device=self.device),
                torch.tensor(abs(self.vx_range[0]), device=self.device),
            )
            ratio_sq = (vx.abs() / limit_x.clamp(min=1e-6)).pow(2).clamp(0.0, 1.0)
            max_vy = self.vy_range[1] * torch.sqrt(1.0 - ratio_sq)
            vy = (torch.rand(n, device=self.device) * 2.0 - 1.0) * max_vy

        self.vel_cmd[env_ids, 0] = vx
        self.vel_cmd[env_ids, 1] = vy

        # EE targets
        if self.fixed_ee_targets is not None:
            for i, (tx, ty, tz) in enumerate(self.fixed_ee_targets[: self.num_arm_ee]):
                self.ee_target_body[env_ids, i] = torch.tensor(
                    [tx, ty, tz], device=self.device, dtype=torch.float32
                )
        else:
            for i in range(self.num_arm_ee):
                y_lo, y_hi = self._arm_y_ranges[i]
                self.ee_target_body[env_ids, i, 0] = torch.empty(n, device=self.device).uniform_(*self.ee_x_range)
                self.ee_target_body[env_ids, i, 1] = torch.empty(n, device=self.device).uniform_(y_lo, y_hi)
                self.ee_target_body[env_ids, i, 2] = torch.empty(n, device=self.device).uniform_(*self.ee_z_range)

        self.cmd_timer[env_ids] = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_arm_ee_pos_body(self) -> torch.Tensor:
        """Current arm EE positions in robot body frame. Shape: (N, num_arm_ee, 3)."""
        root_state = self.robot.data.root_link_state_w
        root_pos_w = root_state[:, :3]
        root_quat = quat_normalize(root_state[:, 3:7])

        ee_pos_w = self.robot.data.body_state_w[:, self.arm_ee_isaac_indices, :3]
        ee_rel_w = ee_pos_w - root_pos_w.unsqueeze(1)

        q_exp = einops.repeat(root_quat, "N d -> N K d", K=self.num_arm_ee)
        return quat_rotate_inv(q_exp, ee_rel_w)  # (N, num_arm_ee, 3)

    # ------------------------------------------------------------------
    # Core env overrides
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        s_cur, _, lin_vel_body, ang_vel_body, _, _, _ = self._get_s_cur()

        ee_pos_b = self._get_arm_ee_pos_body()                    # (N, num_arm_ee, 3)
        ee_error = self.ee_target_body - ee_pos_b                 # (N, num_arm_ee, 3)
        ee_error_flat = ee_error.reshape(self.num_envs, -1)       # (N, num_arm_ee*3)

        g_task = torch.cat([self.vel_cmd, ee_error_flat], dim=-1) # (N, task_goal_dim)
        obs = torch.cat([s_cur, g_task], dim=-1)

        self._cached_lin_vel_body = lin_vel_body
        self._cached_ee_pos_b = ee_pos_b

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        v_body = self._cached_lin_vel_body
        vx_cur, vy_cur = v_body[:, 0], v_body[:, 1]
        vx_cmd, vy_cmd = self.vel_cmd[:, 0], self.vel_cmd[:, 1]

        # Linear velocity tracking
        lin_vel_err = (vx_cur - vx_cmd).pow(2) + (vy_cur - vy_cmd).pow(2)
        r_vel = torch.exp(-self.rew_lin_vel_pen * lin_vel_err)

        # EE reaching (mean squared distance across both arms)
        ee_pos_b = self._cached_ee_pos_b                              # (N, num_arm_ee, 3)
        ee_dist_sq = (self.ee_target_body - ee_pos_b).pow(2).sum(-1)  # (N, num_arm_ee)
        ee_dist_mean = ee_dist_sq.mean(-1)                            # (N,)
        r_ee = torch.exp(-self.rew_ee_pen * ee_dist_mean)

        reward = self.rew_w_lin_vel * r_vel + self.rew_w_ee_reach * r_ee

        # Command resampling
        self.cmd_timer += 1
        resample_mask = self.cmd_timer >= self.cmd_resample_steps
        if resample_mask.any():
            resample_ids = resample_mask.nonzero(as_tuple=False).squeeze(-1)
            self._resample_commands(resample_ids)

        with torch.no_grad():
            self._dbg_lin_vel_err = lin_vel_err
            self._dbg_ee_dist = ee_dist_sq.mean(-1).sqrt()

        return reward

    def _get_dones(self):
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

        # Reference-State Initialisation
        min_episode_frames = 60
        max_start = max(0, self.max_frame_idx - min_episode_frames)
        random_starts = torch.randint(0, max_start + 1, (n,), device=self.device)
        self.ref_frame_idx[env_ids] = random_starts

        root_pos_0          = self._ref_root_pos[random_starts]
        root_quat_0         = self._ref_root_quat_wxyz[random_starts]
        root_lin_vel_0_body = self._ref_root_lin_vel[random_starts].clone()
        root_ang_vel_0_body = self._ref_root_ang_vel[random_starts].clone()
        joints_0            = self._ref_joints[random_starts]
        joint_vel_0         = self._ref_joint_vel[random_starts]

        # Random heading (same as navigation task for generalisation)
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
        w_w      = quat_rotate(root_quat_w, root_ang_vel_0_body)
        r_body   = self.robot.data.body_com_pos_b[env_ids, 0]
        r_w      = quat_rotate(root_quat_w, r_body)
        v_com_w  = v_link_w + torch.linalg.cross(w_w, r_w, dim=-1)

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
        self.robot.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, None, env_ids
        )

        self._resample_commands(env_ids)
        self._cached_ref_tensors = None

    def step(self, action):
        obs, rew, terminated, truncated, extras = super().step(action)

        done = terminated | truncated
        if not done.any():
            return obs, rew, terminated, truncated, extras

        extras = extras or {}
        log = extras.setdefault("log", {})
        ids = done.nonzero(as_tuple=False).squeeze(-1)

        def _mean(x):
            return x.index_select(0, ids).mean().item()

        if self._dbg_lin_vel_err is not None:
            log["lin_vel_err_mean"] = _mean(self._dbg_lin_vel_err)
        if self._dbg_ee_dist is not None:
            log["ee_dist_mean_m"] = _mean(self._dbg_ee_dist)
        if self._dbg_fallen is not None:
            log["fallen_pct"] = _mean(self._dbg_fallen.float()) * 100.0

        return obs, rew, terminated, truncated, extras
