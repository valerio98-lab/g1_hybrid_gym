# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
from g1_hybrid_prior.dataset import G1HybridPriorDataset


class G1HybridGymEnv(DirectRLEnv):
    cfg: G1HybridGymEnvCfg

    def __init__(
        self, cfg: G1HybridGymEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.dataset = G1HybridPriorDataset(
            file_path=Path(
                "/home/valerio/g1_hybrid_prior/data_raw/LAFAN1_Retargeting_Dataset/g1/dance1_subject2.csv"
            ),
            robot="g1",
            lazy_load=False,
        )

        print(f"[G1HybridGymEnv] Loaded dataset with {len(self.dataset)} frames")

        dataset_joint_names = self.dataset.robot_cfg.joint_order
        print("[G1HybridGymEnv] Dataset joints:", dataset_joint_names)

        isaac_joint_names = (
            self.robot.joint_names
        )  ## robot joint names in isaac order. The function returns a list of str
        print("[G1HybridGymEnv] Isaac joints:", isaac_joint_names)

        dataset_to_isaac_indexes: list[int] = []
        _g1_dof_idx = []
        for name in dataset_joint_names:
            if name not in isaac_joint_names:
                raise ValueError(
                    f"[G1HybridGymEnv] Joint '{name}' from dataset not found in robot articulation!"
                )
            dataset_to_isaac_indexes.append(isaac_joint_names.index(name))
            joint_idx, _ = self.robot.find_joints(name)
            _g1_dof_idx.append(joint_idx)
        
        self.dataset_to_isaac_indexes = torch.tensor(
            dataset_to_isaac_indexes, device=self.device
        )
        self._g1_dof_idx = torch.tensor(_g1_dof_idx, device=self.device)
            

        self.ref_frame_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.max_frame_idx = len(self.dataset) - 1

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.actions = None

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is not None:
            self.actions = actions.clone()


    def _apply_action(self) -> None:
        ref_joint_pos = []
        ref_joint_vel = []
        for env_id in range(self.num_envs):
            frame_idx = int(self.ref_frame_idx[env_id].item())
            frame = self.dataset[frame_idx]
            ref_joint_pos.append(frame["joints"])
            ref_joint_vel.append(frame["joint_vel"])

        ref_joint_pos = torch.stack(ref_joint_pos, dim=0).to(self.device).unsqueeze(-1)
        ref_joint_vel = torch.stack(ref_joint_vel, dim=0).to(self.device).unsqueeze(-1)
        
        # apply action as PD target
        self.robot.set_joint_position_target(ref_joint_pos, joint_ids=self._g1_dof_idx)
        self.robot.set_joint_velocity_target(ref_joint_vel, joint_ids=self._g1_dof_idx)

    def _get_observations(self) -> dict:
        joint_full_pos = self.robot.data.joint_pos
        joint_full_vel = self.robot.data.joint_vel
        joint_pos = joint_full_pos[:, self.dataset_to_isaac_indexes]
        joint_vel = joint_full_vel[:, self.dataset_to_isaac_indexes]

        ref_joint_pos = []
        ref_joint_vel = []

        for env_id in range(self.num_envs):
            frame_idx = int(self.ref_frame_idx[env_id].item())
            frame = self.dataset[frame_idx]
            ref_joint_pos.append(frame["joints"])
            ref_joint_vel.append(frame["joint_vel"])

        ref_joint_pos = torch.stack(ref_joint_pos, dim=0).to(self.device)
        ref_joint_vel = torch.stack(ref_joint_vel, dim=0).to(self.device)

        obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                ref_joint_pos,
                ref_joint_vel,
            ),
            dim=-1,
        )
        alive = ~self.reset_buf
        self.ref_frame_idx[alive] += 1
        self.ref_frame_idx.clamp_(0, self.max_frame_idx)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]
        joint_vel = self.robot.data.joint_vel[:, self.dataset_to_isaac_indexes]
        ref_joint_pos = []
        ref_joint_vel = []
        for env_id in range(self.num_envs):
            frame_idx = int(self.ref_frame_idx[env_id].item())
            frame = self.dataset[frame_idx]
            ref_joint_pos.append(frame["joints"])
            ref_joint_vel.append(frame["joint_vel"])

        ref_joint_pos = torch.stack(ref_joint_pos, dim=0).to(self.device)
        ref_joint_vel = torch.stack(ref_joint_vel, dim=0).to(self.device)

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
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = torch.zeros_like(time_out, dtype=torch.bool, device=self.device)

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)

        device = self.device
        self.ref_frame_idx[env_ids] = 0 

        # stato di default
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # origins delle env
        env_origins = self.scene.env_origins[env_ids]
        # applica frame LAFAN a root + joint
        frame = self.dataset[0]
        for i in range(env_ids.shape[0]):
            # frame_idx = int(self.ref_frame_idx[env_id].item())
            # estraggo root pose, orient e vel dal dataset
            root_pos = frame["root_pos"].to(device=device, dtype=torch.float32)
            root_quat = frame["root_quat_wxyz"].to(device=device, dtype=torch.float32)

            root_lin_vel = frame["root_lin_vel"].to(device=device, dtype=torch.float32)
            root_ang_vel = frame["root_ang_vel"].to(device=device, dtype=torch.float32)

            ##Idem per joints
            joints = frame["joints"].to(device=device, dtype=torch.float32)
            joint_vel = frame["joint_vel"].to(device=device, dtype=torch.float32)

            # Sovrascrivo la root pos/rot di default con root pos + offsettata dall'origine dell'env
            default_root_state[i, 0:3] = root_pos + env_origins[i]
            default_root_state[i, 3:7] = root_quat

            # root vel
            default_root_state[i, 7:10] = root_lin_vel
            default_root_state[i, 10:13] = root_ang_vel

            # joints solo sui DOF tracciati
            default_joint_pos[i, self.dataset_to_isaac_indexes] = joints
            default_joint_vel[i, self.dataset_to_isaac_indexes] = joint_vel

        # applica allo stato della simulazione
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, None, env_ids
        )



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
    rew_alive = rew_alive * torch.ones_like(rew_pose)
    rew_alive_term = rew_alive * (1.0 - reset_terminated.float())
    total_reward = rew_pose + rew_vel + rew_alive_term

    return total_reward
