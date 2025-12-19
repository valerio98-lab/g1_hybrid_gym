# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import Dict
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
    quat_mul,
    quat_inv,
    wrap_to_pi,
)


class G1HybridGymEnv(DirectRLEnv):
    cfg: G1HybridGymEnvCfg

    def __init__(
        self, cfg: G1HybridGymEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        dataset_path = Path("/home/valerio/g1_hybrid_prior/data_raw/LAFAN1_Retargeting_Dataset/g1_ee_augmented")
        self.dataset = G1HybridPriorDataset(
            file_path=dataset_path,
            robot="g1",
            dataset_type="augmented",
            lazy_load=False,  
            vel_mode="central",
        )
        print(f"[G1HybridGymEnv] Loaded AUGMENTED dataset with {len(self.dataset)} frames")

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
            g1_dof_idx.append(int(joint_idx_t.item()))

        self.dataset_to_isaac_indexes = torch.tensor(
            dataset_to_isaac_indexes, device=self.device, dtype=torch.long
        )

        # Keep (J,1) on purpose -> targets must be (N,J,1)
        self._g1_dof_idx = torch.tensor(
            g1_dof_idx, device=self.device, dtype=torch.long
        ).unsqueeze(-1)

        # --- End-Effector Mapping (Isaac Indices) ---
        # Cerchiamo gli indici dei body in Isaac che corrispondono agli EE del dataset
        self.ee_names = self.dataset.robot_cfg.ee_link_names
        self.ee_isaac_indices = []
        
        all_body_names = self.robot.body_names
        for ee_name in self.ee_names:
            if ee_name not in all_body_names:
                raise ValueError(f"End-Effector Link '{ee_name}' not found in Isaac articulation bodies!")
            self.ee_isaac_indices.append(all_body_names.index(ee_name))
            
        self.ee_isaac_indices = torch.tensor(self.ee_isaac_indices, device=self.device, dtype=torch.long)
        print(f"[G1HybridGymEnv] Mapped End-Effectors: {self.ee_names} -> Indices {self.ee_isaac_indices.tolist()}")

        # Reference index per env
        self.ref_frame_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.max_frame_idx = len(self.dataset) - 1

        self.actions: torch.Tensor | None = None

        # Caching reference tensors
        self._cached_ref_tensors: Dict[str, torch.Tensor] | None = None

        # Pre-stacked reference tensors
        self._ref_root_pos: torch.Tensor | None = None
        self._ref_root_quat_wxyz: torch.Tensor | None = None
        self._ref_root_lin_vel: torch.Tensor | None = None
        self._ref_root_ang_vel: torch.Tensor | None = None
        self._ref_joints: torch.Tensor | None = None
        self._ref_joint_vel: torch.Tensor | None = None
        self._ref_ee_pos: torch.Tensor | None = None  # Nuovo tensore per EE

        self._build_reference_tensors()

    def _build_reference_tensors(self) -> None:
        """Pre-stack dataset frames into tensors for fast batched indexing."""
        if getattr(self.dataset, "lazy_load", False):
            return

        frames = getattr(self.dataset, "dataset", None)
        if not frames:
            return

        root_pos = torch.stack([f["root_pos"] for f in frames], dim=0)
        root_quat = torch.stack([f["root_quat_wxyz"] for f in frames], dim=0)
        root_lin_vel = torch.stack([f["root_lin_vel"] for f in frames], dim=0)
        root_ang_vel = torch.stack([f["root_ang_vel"] for f in frames], dim=0)
        joints = torch.stack([f["joints"] for f in frames], dim=0)
        joint_vel = torch.stack([f["joint_vel"] for f in frames], dim=0)

        self._ref_root_pos = root_pos.to(self.device)
        self._ref_root_quat_wxyz = root_quat.to(self.device)
        self._ref_root_lin_vel = root_lin_vel.to(self.device)
        self._ref_root_ang_vel = root_ang_vel.to(self.device)
        self._ref_joints = joints.to(self.device)
        self._ref_joint_vel = joint_vel.to(self.device)
        
        # Stack EE pos (Se presente nel dataset)
        if "ee_pos" in frames[0]:
            self._ref_ee_pos = torch.stack([f["ee_pos"] for f in frames], dim=0).to(self.device)

    def _get_ref_batch(self, frame_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a batched reference dict for given per-env frame indices."""
        idx = frame_idx.clamp(0, self.max_frame_idx)

        batch = {
            "root_pos": self._ref_root_pos.index_select(0, idx),
            "root_quat_wxyz": self._ref_root_quat_wxyz.index_select(0, idx),
            "root_lin_vel": self._ref_root_lin_vel.index_select(0, idx),
            "root_ang_vel": self._ref_root_ang_vel.index_select(0, idx),
            "joints": self._ref_joints.index_select(0, idx),
            "joint_vel": self._ref_joint_vel.index_select(0, idx),
        }
        
        # Aggiungi EE se disponibile
        if self._ref_ee_pos is not None:
            batch["ee_pos"] = self._ref_ee_pos.index_select(0, idx)
            
        return batch

    def _build_goal_from_ref(
        self,
        h_cur: torch.Tensor,
        q_cur_wxyz: torch.Tensor,
        v_cur_body: torch.Tensor,
        w_cur_body: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        ref: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calcola l'errore tra stato corrente e riferimento (osservazione)."""
        # --- height error ---
        h_ref = ref["root_pos"][:, 2:3]
        dh = h_ref - h_cur

        # --- orientation error ---
        q_ref = quat_normalize(ref["root_quat_wxyz"])
        q_cur = quat_normalize(q_cur_wxyz)
        q_err = quat_mul(q_ref, quat_inv(q_cur))
        q_err = quat_normalize(q_err)

        # --- velocities ---
        v_ref_body_ref = ref["root_lin_vel"]
        w_ref_body_ref = ref["root_ang_vel"]
        
        v_ref_world = quat_rotate(q_ref, v_ref_body_ref)
        w_ref_world = quat_rotate(q_ref, w_ref_body_ref)

        v_ref_body_sim = quat_rotate_inv(q_cur, v_ref_world)
        w_ref_body_sim = quat_rotate_inv(q_cur, w_ref_world)

        dv = v_ref_body_sim - v_cur_body
        dw = w_ref_body_sim - w_cur_body

        # --- joints ---
        dq = wrap_to_pi(ref["joints"] - joint_pos)
        dqd = ref["joint_vel"] - joint_vel

        return torch.cat((dh, q_err, dv, dw, dq, dqd), dim=-1)

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

        joint_pos = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]
        target_joint_pos = joint_pos + scaled_actions
        target_joint_pos = target_joint_pos.unsqueeze(-1)

        self.robot.set_joint_position_target(
            target_joint_pos, joint_ids=self._g1_dof_idx
        )

    def _get_observations(self) -> dict:
        root_link_state = self.robot.data.root_link_state_w  # (N,13)
        root_pos_w = root_link_state[:, 0:3]
        root_quat_wxyz = quat_normalize(root_link_state[:, 3:7])  # (N,4)

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

        self.ref_frame_idx.clamp_(0, self.max_frame_idx)
        ref = self._get_ref_batch(self.ref_frame_idx)
        self._cached_ref_tensors = ref

        goal = self._build_goal_from_ref(
            h_cur=h,
            q_cur_wxyz=root_quat_wxyz,
            v_cur_body=root_lin_vel_body,
            w_cur_body=root_ang_vel_body,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            ref=ref,
        )

        obs = torch.cat((s_cur, goal), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Dati Robot Correnti
        joint_pos = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]
        joint_vel = self.robot.data.joint_vel[:, self.dataset_to_isaac_indexes]
        
        # Root state relativo all'origine dell'env
        root_link_state = self.robot.data.root_link_state_w
        root_pos_w = root_link_state[:, 0:3] - self.scene.env_origins 
        root_quat_w = quat_normalize(root_link_state[:, 3:7])

        # End-Effector State Corrente (Robot)
        # body_state_w: (num_envs, num_bodies, 13)
        # Estraiamo solo i body delle mani/piedi usando gli indici mappati
        # Risultato: (N, 4, 3)
        ee_state_w = self.robot.data.body_state_w[:, self.ee_isaac_indices, 0:3]
        # Shiftiamo anche questo all'origine dell'env per confrontarlo col dataset
        # env_origins è (N, 3), lo espandiamo a (N, 1, 3) per broadcasting
        ee_pos_rel = ee_state_w - self.scene.env_origins.unsqueeze(1)

        # Dati Reference
        if self._cached_ref_tensors is not None:
            ref = self._cached_ref_tensors
        else:
            self.ref_frame_idx.clamp_(0, self.max_frame_idx)
            ref = self._get_ref_batch(self.ref_frame_idx)

        total_reward = compute_rewards(
            self.cfg.rew_w_pose,
            self.cfg.rew_w_vel,
            self.cfg.rew_w_root_pos,
            self.cfg.rew_w_root_rot,
            self.cfg.rew_w_ee,
            self.cfg.rew_alive,
            # Current Robot State
            joint_pos, joint_vel, root_pos_w, root_quat_w, ee_pos_rel,
            # Reference State
            ref["joints"], ref["joint_vel"], ref["root_pos"], ref["root_quat_wxyz"], ref.get("ee_pos"),
            # Flags
            self.reset_terminated,
        )

        alive = ~self.reset_buf
        self.ref_frame_idx[alive] += 1
        overflow = self.ref_frame_idx > self.max_frame_idx
        self.ref_frame_idx[overflow] = 0

        self._cached_ref_tensors = None
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
        n = len(env_ids)

        # --- RANDOM START (RSI) ---
        min_episode_frames = 60
        max_start = max(0, self.max_frame_idx - min_episode_frames)
        random_starts = torch.randint(0, max_start + 1, (n,), device=self.device)
        self.ref_frame_idx[env_ids] = random_starts

        if self._ref_root_pos is not None:
            root_pos_0 = self._ref_root_pos[random_starts]
            root_quat_0 = self._ref_root_quat_wxyz[random_starts]
            root_lin_vel_0_body = self._ref_root_lin_vel[random_starts]
            root_ang_vel_0_body = self._ref_root_ang_vel[random_starts]
            joints_0 = self._ref_joints[random_starts]
            joint_vel_0 = self._ref_joint_vel[random_starts]
        else:
            # Fallback lazy load
            frame0 = self.dataset[0] 
            root_pos_0 = frame0["root_pos"].to(self.device).repeat(n, 1)
            # (logica lazy completa omessa per brevità dato che usi pre-stack)

        env_origins = self.scene.env_origins[env_ids]
        root_pos_w = root_pos_0 + env_origins
        root_quat_w = quat_normalize(root_quat_0)

        # Velocità: Body -> World -> COM
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


@torch.jit.script
def compute_rewards(
    rew_w_pose: float,
    rew_w_vel: float,
    rew_w_root_pos: float,
    rew_w_root_rot: float,
    rew_w_ee: float,
    rew_alive: float,
    # Current
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    ee_pos: torch.Tensor,
    # Ref
    ref_joint_pos: torch.Tensor,
    ref_joint_vel: torch.Tensor,
    ref_root_pos: torch.Tensor,
    ref_root_quat: torch.Tensor,
    ref_ee_pos: torch.Tensor | None,
    # Flags
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    
    # 1. Calcolo Errori Quadratici (L2^2)
    # Pose joints
    joint_pos_err = torch.sum(torch.square(joint_pos - ref_joint_pos), dim=-1)
    # Velocity joints
    joint_vel_err = torch.sum(torch.square(joint_vel - ref_joint_vel), dim=-1)
    
    # Root position
    root_pos_diff = root_pos - ref_root_pos
    root_pos_err = torch.sum(torch.square(root_pos_diff), dim=-1)
    
    # Root rotation (1 - |dot|)
    quat_dot = torch.sum(root_quat * ref_root_quat, dim=-1).abs().clamp(0.0, 1.0)
    root_rot_err = 1.0 - quat_dot

    # End-Effector Error
    if ref_ee_pos is not None:
        ee_diff = ee_pos - ref_ee_pos
        # Errore quadro per ogni EE, sommato su XYZ -> (N, 4)
        ee_sq_err = torch.sum(torch.square(ee_diff), dim=-1)
        # Somma su tutti gli EE -> (N,)
        ee_total_err = torch.sum(ee_sq_err, dim=-1)
    else:
        ee_total_err = torch.zeros_like(root_pos_err)

    # 2. Kernel Esponenziali (Moltiplicativi)
    # Usiamo i pesi passati dal config come coefficienti negativi per l'esponenziale
    # Formula: exp(-weight * error)
    
    r_pose   = torch.exp(-rew_w_pose * joint_pos_err)
    r_vel    = torch.exp(-rew_w_vel * joint_vel_err)
    r_root_p = torch.exp(-rew_w_root_pos * root_pos_err)
    r_root_r = torch.exp(-rew_w_root_rot * root_rot_err)
    r_ee     = torch.exp(-rew_w_ee * ee_total_err)

    # 3. Reward Totale
    # Struttura moltiplicativa per l'imitazione: forza il rispetto di tutti i vincoli
    imitation_reward = r_pose * r_vel * r_root_p * r_root_r * r_ee
    
    # Bonus sopravvivenza additivo (per gradienti stabili all'inizio)
    alive_bonus = rew_alive * (1.0 - reset_terminated.float())

    return imitation_reward + alive_bonus