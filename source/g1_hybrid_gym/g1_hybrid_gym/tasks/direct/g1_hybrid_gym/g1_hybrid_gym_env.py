# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
from g1_hybrid_prior.dataset import G1HybridPriorDataset
from g1_hybrid_prior.helpers import (
    quat_normalize,
    quat_rotate,
    quat_rotate_inv,
)


class G1HybridGymEnv(DirectRLEnv):
    cfg: G1HybridGymEnvCfg

    def __init__(
        self, cfg: G1HybridGymEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Keep previous orientation to express CURRENT sim velocities in (prev) BODY frame
        # (matches your dataset convention)
        self._prev_root_quat_wxyz: torch.Tensor | None = None  # (N,4)

        # --- Dataset ---
        self.dataset = G1HybridPriorDataset(
            file_path=Path(
                "/home/valerio/g1_hybrid_prior/data_raw/LAFAN1_Retargeting_Dataset/g1/dance1_subject2.csv"
            ),
            robot="g1",
            lazy_load=False,
        )
        print(f"[G1HybridGymEnv] Loaded dataset with {len(self.dataset)} frames")

        dataset_joint_names = self.dataset.robot_cfg.joint_order
        isaac_joint_names = self.robot.joint_names

        dataset_to_isaac_indexes: list[int] = []
        g1_dof_idx: list[int] = []

        for name in dataset_joint_names:
            if name not in isaac_joint_names:
                raise ValueError(
                    f"[G1HybridGymEnv] Joint '{name}' from dataset not found in robot articulation!"
                )

            dataset_to_isaac_indexes.append(isaac_joint_names.index(name))

            joint_idx, _ = self.robot.find_joints(name)
            joint_idx_t = (
                torch.as_tensor(joint_idx, dtype=torch.long).reshape(-1).to(self.device)
            )
            if joint_idx_t.numel() != 1:
                raise RuntimeError(
                    f"find_joints('{name}') returned {joint_idx_t.numel()} indices: {joint_idx_t}"
                )

            g1_dof_idx.append(int(joint_idx_t.item()))

        self.dataset_to_isaac_indexes = torch.tensor(
            dataset_to_isaac_indexes, device=self.device, dtype=torch.long
        )

        # Keep (J,1) on purpose -> targets must be (N,J,1)
        self._g1_dof_idx = torch.tensor(
            g1_dof_idx, device=self.device, dtype=torch.long
        ).unsqueeze(-1)
        assert self._g1_dof_idx.ndim == 2 and self._g1_dof_idx.shape[1] == 1, (
            f"Expected _g1_dof_idx shape (J,1), got {tuple(self._g1_dof_idx.shape)}. "
            "If you switch to (J,), remove the unsqueeze(-1) in _apply_action."
        )

        # Reference index per env
        self.ref_frame_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.max_frame_idx = len(self.dataset) - 1

        self.actions: torch.Tensor | None = None

        # Cache to guarantee reward uses the *exact same ref* as obs in the step
        self._cached_ref_joint_pos: torch.Tensor | None = None
        self._cached_ref_joint_vel: torch.Tensor | None = None

        # Pre-stacked reference tensors for fast indexing (if non-lazy)
        self._ref_root_pos: torch.Tensor | None = None
        self._ref_root_quat_wxyz: torch.Tensor | None = None
        self._ref_root_lin_vel: torch.Tensor | None = None
        self._ref_root_ang_vel: torch.Tensor | None = None
        self._ref_joints: torch.Tensor | None = None
        self._ref_joint_vel: torch.Tensor | None = None

        self._build_reference_tensors()

    # -------------------------------------------------------------------------
    # Reference utils
    # -------------------------------------------------------------------------

    def _build_reference_tensors(self) -> None:
        """Pre-stack dataset frames into tensors for fast batched indexing."""
        if getattr(self.dataset, "lazy_load", False):
            return

        frames = getattr(self.dataset, "dataset", None)
        if not frames:
            return

        root_pos = torch.stack([f["root_pos"] for f in frames], dim=0)  # (T,3)
        root_quat = torch.stack([f["root_quat_wxyz"] for f in frames], dim=0)  # (T,4)
        root_lin_vel = torch.stack(
            [f["root_lin_vel"] for f in frames], dim=0
        )  # (T,3) BODY
        root_ang_vel = torch.stack(
            [f["root_ang_vel"] for f in frames], dim=0
        )  # (T,3) BODY
        joints = torch.stack([f["joints"] for f in frames], dim=0)  # (T,J)
        joint_vel = torch.stack([f["joint_vel"] for f in frames], dim=0)  # (T,J)

        self._ref_root_pos = root_pos.to(self.device)
        self._ref_root_quat_wxyz = root_quat.to(self.device)
        self._ref_root_lin_vel = root_lin_vel.to(self.device)
        self._ref_root_ang_vel = root_ang_vel.to(self.device)
        self._ref_joints = joints.to(self.device)
        self._ref_joint_vel = joint_vel.to(self.device)

    def _get_ref_batch(self, frame_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a batched reference dict for given per-env frame indices."""
        idx = frame_idx.clamp(0, self.max_frame_idx)

        if self._ref_joints is not None:
            return {
                "root_pos": self._ref_root_pos.index_select(0, idx),
                "root_quat_wxyz": self._ref_root_quat_wxyz.index_select(0, idx),
                "root_lin_vel": self._ref_root_lin_vel.index_select(0, idx),
                "root_ang_vel": self._ref_root_ang_vel.index_select(0, idx),
                "joints": self._ref_joints.index_select(0, idx),
                "joint_vel": self._ref_joint_vel.index_select(0, idx),
            }

        # Fallback (lazy loading / no pre-stack)
        ref_joint_pos, ref_joint_vel = [], []
        ref_root_pos, ref_root_quat = [], []
        ref_root_lin_vel, ref_root_ang_vel = [], []

        for env_id in range(self.num_envs):
            fi = int(idx[env_id].item())
            frame = self.dataset[fi]
            ref_joint_pos.append(frame["joints"])
            ref_joint_vel.append(frame["joint_vel"])
            ref_root_pos.append(frame["root_pos"])
            ref_root_quat.append(frame["root_quat_wxyz"])
            ref_root_lin_vel.append(frame["root_lin_vel"])
            ref_root_ang_vel.append(frame["root_ang_vel"])

        return {
            "root_pos": torch.stack(ref_root_pos, dim=0).to(self.device),
            "root_quat_wxyz": torch.stack(ref_root_quat, dim=0).to(self.device),
            "root_lin_vel": torch.stack(ref_root_lin_vel, dim=0).to(self.device),
            "root_ang_vel": torch.stack(ref_root_ang_vel, dim=0).to(self.device),
            "joints": torch.stack(ref_joint_pos, dim=0).to(self.device),
            "joint_vel": torch.stack(ref_joint_vel, dim=0).to(self.device),
        }

    # -------------------------------------------------------------------------
    # IsaacLab hooks
    # -------------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is not None:
            self.actions = actions.clone()

    def _apply_action(self) -> None:
        if self.actions is None:
            return

        actions = self.actions.to(self.device)
        scaled_actions = actions * self.cfg.action_scale

        joint_pos = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]  # (N,J)
        target_joint_pos = joint_pos + scaled_actions  # (N,J)

        # joint_ids is (J,1) => Isaac expects target (N,J,1)
        target_joint_pos = target_joint_pos.unsqueeze(-1)

        self.robot.set_joint_position_target(
            target_joint_pos, joint_ids=self._g1_dof_idx
        )

    def _get_observations(self) -> dict:
        # IMPORTANT:
        # - root_state_w is HYBRID (pose = link, vel = COM) -> NOT what we want.
        # - root_link_state_w is CONSISTENT (pose+vel = link actor frame, all in WORLD).
        root_link_state = self.robot.data.root_link_state_w  # (N,13)
        root_pos_w = root_link_state[:, 0:3]
        root_quat_wxyz = quat_normalize(root_link_state[:, 3:7])  # (N,4) wxyz

        v_link_world = root_link_state[:, 7:10]  # (N,3) WORLD
        w_link_world = root_link_state[:, 10:13]  # (N,3) WORLD

        # Initialize "prev quat" buffer at first call
        if self._prev_root_quat_wxyz is None:
            self._prev_root_quat_wxyz = root_quat_wxyz.clone()

        # Express current sim velocities in (prev) BODY axes to match dataset convention
        q_prev = quat_normalize(self._prev_root_quat_wxyz)
        root_lin_vel_body = quat_rotate_inv(q_prev, v_link_world)  # (N,3) BODY(prev)
        root_ang_vel_body = quat_rotate_inv(q_prev, w_link_world)  # (N,3) BODY(prev)

        # Update prev orientation for next step
        self._prev_root_quat_wxyz = root_quat_wxyz.clone()

        # Height wrt env origin
        env_origins = self.scene.env_origins
        h = (root_pos_w - env_origins)[:, 2:3]  # (N,1)

        # Joint state in dataset order
        joint_pos = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]
        joint_vel = self.robot.data.joint_vel[:, self.dataset_to_isaac_indexes]

        # Current state (cur): BODY velocities + quaternion wxyz
        s_cur = torch.cat(
            (
                h,
                root_quat_wxyz,
                root_lin_vel_body,
                root_ang_vel_body,
                joint_pos,
                joint_vel,
            ),
            dim=-1,
        )

        # Reference state (ref): dataset already stores BODY velocities
        self.ref_frame_idx.clamp_(0, self.max_frame_idx)
        ref = self._get_ref_batch(self.ref_frame_idx)

        self._cached_ref_joint_pos = ref["joints"]
        self._cached_ref_joint_vel = ref["joint_vel"]

        h_ref = ref["root_pos"][:, 2:3]
        s_ref = torch.cat(
            (
                h_ref,
                ref["root_quat_wxyz"],
                ref["root_lin_vel"],
                ref["root_ang_vel"],
                ref["joints"],
                ref["joint_vel"],
            ),
            dim=-1,
        )

        obs = torch.cat((s_cur, s_ref), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]
        joint_vel = self.robot.data.joint_vel[:, self.dataset_to_isaac_indexes]

        if (
            self._cached_ref_joint_pos is not None
            and self._cached_ref_joint_vel is not None
        ):
            ref_joint_pos = self._cached_ref_joint_pos
            ref_joint_vel = self._cached_ref_joint_vel
        else:
            self.ref_frame_idx.clamp_(0, self.max_frame_idx)
            ref = self._get_ref_batch(self.ref_frame_idx)
            ref_joint_pos = ref["joints"]
            ref_joint_vel = ref["joint_vel"]

        total_reward = compute_rewards(
            self.cfg.rew_w_pose,
            self.cfg.rew_w_vel,
            self.cfg.rew_alive,
            joint_pos,
            joint_vel,
            ref_joint_pos,
            ref_joint_vel,
            self.reset_terminated,
        )

        # Advance reference index for envs still alive
        alive = ~self.reset_buf
        self.ref_frame_idx[alive] += 1
        overflow = self.ref_frame_idx > self.max_frame_idx
        self.ref_frame_idx[overflow] = 0

        # Clear cache (rebuilt next obs)
        self._cached_ref_joint_pos = None
        self._cached_ref_joint_vel = None

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = self.robot.data.root_link_state_w[:, 2] < self.cfg.min_height_reset
        terminated = fallen
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)

        # Reset reference timeline for these envs
        self.ref_frame_idx[env_ids] = 0

        default_root_state = self.robot.data.default_root_state[
            env_ids
        ].clone()  # (n,13)
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        env_origins = self.scene.env_origins[env_ids]
        n = env_ids.shape[0]

        # Read frame 0 from dataset (or stacked tensors)
        if self._ref_root_pos is not None:
            root_pos_0 = self._ref_root_pos[0].to(dtype=torch.float32)  # (3,)
            root_quat_0 = quat_normalize(
                self._ref_root_quat_wxyz[0].to(dtype=torch.float32)
            )  # (4,)
            root_lin_vel_0_body = self._ref_root_lin_vel[0].to(
                dtype=torch.float32
            )  # (3,) BODY
            root_ang_vel_0_body = self._ref_root_ang_vel[0].to(
                dtype=torch.float32
            )  # (3,) BODY
            joints_0 = self._ref_joints[0].to(dtype=torch.float32)  # (J,)
            joint_vel_0 = self._ref_joint_vel[0].to(dtype=torch.float32)  # (J,)
        else:
            frame0 = self.dataset[0]
            root_pos_0 = frame0["root_pos"].to(device=self.device, dtype=torch.float32)
            root_quat_0 = quat_normalize(
                frame0["root_quat_wxyz"].to(device=self.device, dtype=torch.float32)
            )
            root_lin_vel_0_body = frame0["root_lin_vel"].to(
                device=self.device, dtype=torch.float32
            )
            root_ang_vel_0_body = frame0["root_ang_vel"].to(
                device=self.device, dtype=torch.float32
            )
            joints_0 = frame0["joints"].to(device=self.device, dtype=torch.float32)
            joint_vel_0 = frame0["joint_vel"].to(
                device=self.device, dtype=torch.float32
            )

        # Expand per-env
        q0 = root_quat_0.view(1, 4).expand(n, 4)  # (n,4)

        # Dataset velocities are BODY and (by your checks) correspond to the LINK point.
        v_link_w = quat_rotate(q0, root_lin_vel_0_body.view(1, 3).expand(n, 3))  # (n,3)
        w_w = quat_rotate(q0, root_ang_vel_0_body.view(1, 3).expand(n, 3))  # (n,3)

        # PhysX "root velocities" are COM velocities.
        # Convert link-point vel -> COM vel:
        # v_link = v_com + ω × (p_link - p_com)  =>  v_com = v_link + ω × (p_com - p_link)
        r_body = self.robot.data.body_com_pos_b[
            env_ids, 0
        ]  # (n,3) = (p_com - p_link) in LINK frame
        r_w = quat_rotate(q0, r_body)  # (n,3) in WORLD
        v_com_w = v_link_w + torch.linalg.cross(w_w, r_w, dim=-1)

        # Write pose (LINK) + COM velocities (WORLD) as expected by IsaacLab buffers
        default_root_state[:, 0:3] = root_pos_0.view(1, 3) + env_origins
        default_root_state[:, 3:7] = q0
        default_root_state[:, 7:10] = v_com_w
        default_root_state[:, 10:13] = w_w

        # Set joints
        default_joint_pos[:, self.dataset_to_isaac_indexes] = joints_0.view(
            1, -1
        ).expand(n, -1)
        default_joint_vel[:, self.dataset_to_isaac_indexes] = joint_vel_0.view(
            1, -1
        ).expand(n, -1)

        # Write to sim
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, None, env_ids
        )

        # Init/update prev-quat buffer for BODY projection
        if self._prev_root_quat_wxyz is None:
            self._prev_root_quat_wxyz = torch.zeros(
                (self.num_envs, 4), device=self.device, dtype=q0.dtype
            )
            # fill with something valid
            self._prev_root_quat_wxyz[:] = quat_normalize(
                self.robot.data.root_link_pose_w[:, 3:7]
            )

        self._prev_root_quat_wxyz[env_ids] = q0


@torch.jit.script
def compute_rewards(
    rew_w_pose: float,
    rew_w_vel: float,
    rew_alive: float,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    ref_joint_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    pose_error = torch.sum(torch.square(joint_pos - ref_joint_pos), dim=-1)
    vel_error = torch.sum(torch.square(joint_vel - ref_joint_vel), dim=-1)

    rew_pose = -rew_w_pose * pose_error
    rew_vel = -rew_w_vel * vel_error

    alive_bonus = rew_alive * torch.ones_like(rew_pose)
    alive_bonus = alive_bonus * (1.0 - reset_terminated.float())

    return rew_pose + rew_vel + alive_bonus
