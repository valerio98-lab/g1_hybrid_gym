# g1_hybrid_gym/envs/g1_hybrid_gym_env_amp.py

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, Sequence
import gymnasium as gym

from .g1_hybrid_gym_env_base import G1HybridGymEnvBase
from g1_hybrid_prior.dataset.dataset_builder import make_dataset
from g1_hybrid_prior.helpers import quat_normalize, quat_rotate_inv


class G1HybridGymEnvAMP(G1HybridGymEnvBase):
    """Env AMP:
    - dataset NPZ style AMP
    - extras["amp_obs"] = s_cur (per discriminatore)
    - early termination paper-style su TUTTI i rigid bodies (reset max)
    """

    def _load_dataset(self):
        cfg_params = self._read_dataset_params()
        cfg_params["training_type"] = "ppo_amp"
        self.cfg_params = cfg_params
        return make_dataset(cfg=cfg_params, device=self.device)

    def __init__(self, *args, **kwargs):
        # Chiamiamo il genitore. Questo scatenerà _build_reference_tensors.
        # Grazie al fix "lazy init" sotto, non crasherà più.
        super().__init__(*args, **kwargs)
        self._num_amp_obs_steps = self.cfg_params.get("num_amp_obs_steps")
        self.max_rot_reset = bool(self.cfg_params.get("max_rot_reset", True))
        J = int(self.dataset.dof_pos.shape[1])  # oppure self.dataset.dof_pos.shape[1]
        self._amp_obs_per_step = 11 + 2 * J  # deve venire 69 se J=29

        assert self._num_amp_obs_steps >= 2

        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, self._amp_obs_per_step),
            device=self.device,
            dtype=torch.float32,
        )
        print(
            f"[G1HybridGymEnvAMP] Loading AMP dataset with {self._num_amp_obs_steps} obs steps...",
            flush=True,
        )
        print(
            f"[G1HybridGymEnvAMP] AMP shape self._num_amp_obs_steps * self._amp_obs_per_step: {self._num_amp_obs_steps} * {self._amp_obs_per_step} = {self._num_amp_obs_steps * self._amp_obs_per_step}",
            flush=True,
        )
        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._num_amp_obs_steps * self._amp_obs_per_step,),
            dtype=np.float32,
        )

        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

    def _setup_body_mapping(self):
        """Calcola il mapping tra i body di Isaac e quelli del Dataset NPZ."""
        # ------------- Body mapping NPZ -> Isaac -------------
        # NPZ names
        if not hasattr(self.dataset, "npz_body_names"):
            # fallback: dataset potrebbe chiamarlo body_names
            npz_body_names = getattr(self.dataset, "body_names", None)
            if npz_body_names is None:
                raise ValueError(
                    "[G1HybridGymEnvAMP] Dataset must expose npz body names."
                )
        else:
            npz_body_names = self.dataset.npz_body_names

        # Isaac names
        isaac_body_names = self.robot.body_names

        name_to_npz = {n: i for i, n in enumerate(npz_body_names)}
        # ------------- EE mapping (NPZ indices) -------------
        # self.ee_names viene dal Base: dataset.robot_cfg.ee_link_names
        ee_npz_indices = []
        if getattr(self, "ee_names", None):
            for ee_name in self.ee_names:
                if ee_name not in name_to_npz:
                    raise ValueError(
                        f"[G1HybridGymEnvAMP] EE link '{ee_name}' not found in NPZ body names!"
                    )
                ee_npz_indices.append(name_to_npz[ee_name])

        self._ee_npz_indices = torch.as_tensor(
            ee_npz_indices, device=self.device, dtype=torch.long
        )
        ee_npz = [npz_body_names[j] for j in self._ee_npz_indices.tolist()]
        if ee_npz != self.ee_names:
            print("[AMP] EE names (env):", self.ee_names)
            print("[AMP] EE names (npz):", ee_npz)
            raise ValueError("[AMP] EE mapping mismatch")

        body_isaac_indices = []
        body_npz_indices = []

        # Tracciamo SOLO i body che esistono in entrambi (di solito saranno tutti)
        for i, isaac_name in enumerate(isaac_body_names):
            if isaac_name in name_to_npz:
                body_isaac_indices.append(i)
                body_npz_indices.append(name_to_npz[isaac_name])

        if len(body_isaac_indices) == 0:
            raise ValueError(
                "[G1HybridGymEnvAMP] No overlapping body names between NPZ and Isaac."
            )

        self._body_isaac_indices = torch.as_tensor(
            body_isaac_indices, device=self.device, dtype=torch.long
        )
        self._body_npz_indices = torch.as_tensor(
            body_npz_indices, device=self.device, dtype=torch.long
        )
        tracked_isaac = [isaac_body_names[i] for i in self._body_isaac_indices.tolist()]
        tracked_npz = [npz_body_names[j] for j in self._body_npz_indices.tolist()]
        if tracked_isaac != tracked_npz:
            # trova il primo mismatch e stampalo bene
            for k, (a, b) in enumerate(zip(tracked_isaac, tracked_npz)):
                if a != b:
                    raise ValueError(
                        f"[AMP] Body mapping mismatch at k={k}: isaac='{a}' npz='{b}'"
                    )
            raise ValueError("[AMP] Body mapping mismatch (length/order)")

        # Contact bodies (da escludere nel maxdist)
        contact_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        contact_body_ids_local = []
        tracked_names = [isaac_body_names[i] for i in body_isaac_indices]
        name_to_local = {n: i for i, n in enumerate(tracked_names)}
        for n in contact_names:
            if n in name_to_local:
                contact_body_ids_local.append(name_to_local[n])

        self._contact_body_ids = torch.tensor(
            contact_body_ids_local, device=self.device, dtype=torch.long
        )

        print(
            f"[G1HybridGymEnvAMP] Tracking bodies K={len(body_isaac_indices)} | "
            f"contact_ids(local)={contact_body_ids_local}"
        )
        upper_names = self.cfg_params.get("upper_names")
        upper_ids_local = [name_to_local[n] for n in upper_names if n in name_to_local]
        self._upper_body_ids = torch.tensor(
            upper_ids_local, device=self.device, dtype=torch.long
        )

    def _build_reference_tensors(self):
        """Zero-copy: punta ai tensori del NPZ dataset + costruisce ref body pos per reset max."""
        if self.dataset is None:
            return

        if not hasattr(self, "_body_npz_indices"):
            self._setup_body_mapping()

        # Root (body 0 nel NPZ -> già estratto nel dataset)
        self._ref_root_pos = self.dataset.root_pos_w
        self._ref_root_quat_wxyz = self.dataset.root_rot_w

        root_rot = quat_normalize(self.dataset.root_rot_w)
        self._ref_root_lin_vel = quat_rotate_inv(root_rot, self.dataset.root_lin_vel_w)
        self._ref_root_ang_vel = quat_rotate_inv(root_rot, self.dataset.root_ang_vel_w)

        self._ref_joints = self.dataset.dof_pos
        self._ref_joint_vel = self.dataset.dof_vel

        # Bodies positions per early termination (WORLD, relative to env origin == ok)
        # dataset deve esporre body_pos_w: (T, B, 3)
        body_pos_w = getattr(self.dataset, "body_pos_w", None)
        if body_pos_w is None:
            # fallback naming
            body_pos_w = getattr(self.dataset, "body_positions_w", None)
        if body_pos_w is None:
            raise ValueError(
                "[G1HybridGymEnvAMP] Dataset must expose body_pos_w tensor (T,B,3)."
            )

        body_rot_w = getattr(self.dataset, "body_rot_w", None)
        if body_rot_w is None:
            body_rot_w = getattr(self.dataset, "body_rotations_w", None)
        if body_rot_w is None:
            # se nel dataset hai solo raw, aggiungilo in G1AMPDataset come tensor self.body_rot_w
            raise ValueError("Dataset must expose body_rot_w (T,B,4).")

        self._ref_body_rot = body_rot_w.index_select(
            1, self._body_npz_indices
        )  # (T,K,4)

        # EE positions for reward (same as PPO pipeline)
        # body_pos_w: (T, B, 3) in NPZ body order
        # we select only the EE links in the same order as self.ee_names
        if self.ee_isaac_indices is not None and self._ee_npz_indices.numel() > 0:
            self._ref_ee_pos = body_pos_w.index_select(1, self._ee_npz_indices)
        else:
            self._ref_ee_pos = None

        # Reorder/select bodies to match tracked Isaac bodies
        self._ref_body_pos = body_pos_w.index_select(1, self._body_npz_indices)

    def _compute_amp_obs_step(self, obs_dict: dict) -> torch.Tensor:
        # obs_dict["policy"] = [s_cur, goal]
        full_obs = obs_dict["policy"]
        half_len = full_obs.shape[-1] // 2
        return full_obs[..., :half_len]  # s_cur

    def step(self, action):
        obs, rew, terminated, truncated, extras = super().step(action)
        if extras is None:
            extras = {}
        extras["amp_obs"] = self._amp_obs_buf.reshape(self.num_envs, -1)

        return obs, rew, terminated, truncated, extras

    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()

        curr = self._compute_amp_obs_step(obs_dict)

        self._amp_obs_buf = torch.roll(self._amp_obs_buf, shifts=1, dims=1)
        self._amp_obs_buf[:, 0] = curr

        return obs_dict

    def _build_s_cur(self, env_ids: torch.Tensor) -> torch.Tensor:
        # root state
        root_link_state = self.robot.data.root_link_state_w[env_ids]  # (n, 13)
        root_pos_w = root_link_state[:, 0:3]
        root_quat_wxyz = quat_normalize(root_link_state[:, 3:7])

        v_link_world = root_link_state[:, 7:10]
        w_link_world = root_link_state[:, 10:13]

        root_lin_vel_body = quat_rotate_inv(root_quat_wxyz, v_link_world)
        root_ang_vel_body = quat_rotate_inv(root_quat_wxyz, w_link_world)

        env_origins = self.scene.env_origins[env_ids]
        h = (root_pos_w - env_origins)[:, 2:3]

        joint_pos = self.robot.data.joint_pos[env_ids][:, self.dataset_to_isaac_indexes]
        joint_vel = self.robot.data.joint_vel[env_ids][:, self.dataset_to_isaac_indexes]

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
        return s_cur  # (n, 69)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 1) fai tutto il reset standard (quello che hai nel base)
        super()._reset_idx(env_ids)

        # 2) normalizza env_ids come fa il base (per essere robusti)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if hasattr(self.robot, "update"):
            self.robot.update(dt=self.cfg.sim.dt)

        s_cur = self._build_s_cur(env_ids)  # (n, 69)

        # Fill history with same frame (NVIDIA init)
        self._amp_obs_buf[env_ids] = s_cur.unsqueeze(1).repeat(
            1, self._num_amp_obs_steps, 1
        )

    def _get_dones(self):
        terminated, time_out = super()._get_dones()

        # ref (già cached dal base)
        ref = self._cached_ref_tensors
        if ref is None:
            self.ref_frame_idx.clamp_(0, self.max_frame_idx)
            ref = self._get_ref_batch(self.ref_frame_idx)
            self._cached_ref_tensors = ref

        # Paper-style "reset max" su rigid bodies
        # curr_rigid_body_pos: (N,K,3) rel to env origin
        curr_body_pos = self.robot.data.body_state_w[:, self._body_isaac_indices, 0:3]
        curr_body_pos_rel = curr_body_pos - self.scene.env_origins.unsqueeze(1)

        goal_body_pos = ref["body_pos"]  # (N,K,3)

        # compute imitation reset max (threshold 0.5) escludendo contact bodies
        reset_buf = self.reset_buf.to(torch.float32)
        progress_buf = self.episode_length_buf.to(torch.float32)

        if self.max_rot_reset:
            curr_body_rot_wxyz = quat_normalize(
                self.robot.data.body_state_w[:, self._body_isaac_indices, 3:7]
            )
            goal_body_rot_wxyz = quat_normalize(ref["body_rot"])

            upper_body_ids = self._upper_body_ids
            _, term_f = compute_imitation_reset_max_posrot_upper(
                reset_buf=reset_buf,
                progress_buf=progress_buf,
                curr_rigid_body_pos=curr_body_pos_rel,
                goal_rigid_body_pos=goal_body_pos,
                curr_rigid_body_rot=curr_body_rot_wxyz,
                goal_rigid_body_rot=goal_body_rot_wxyz,
                contact_body_ids=self._contact_body_ids,
                upper_body_ids=upper_body_ids,
                max_episode_length=float(self.max_episode_length),
                enable_early_termination=True,
                early_term_rot_threshold_deg=float(
                    self.cfg_params.get("early_term_rot_threshold_deg", 90.0)
                ),
                early_termination_dist_threshold=float(
                    self.cfg_params.get("early_termination_dist_threshold", 0.5)
                ),
            )
        else:
            _, term_f = compute_imitation_reset_max(
                reset_buf=reset_buf,
                progress_buf=progress_buf,
                curr_rigid_body_pos=curr_body_pos_rel,
                goal_rigid_body_pos=goal_body_pos,
                contact_body_ids=self._contact_body_ids,
                max_episode_length=float(self.max_episode_length),
                enable_early_termination=True,
                early_termination_dist_threshold=float(
                    self.cfg_params.get("early_termination_dist_threshold", 0.5)
                ),
            )

        term_amp = term_f > 0.5
        terminated = terminated | term_amp

        return terminated, time_out

    def fetch_amp_expert_batch(self, batch_size: int):
        return self.dataset.sample(batch_size)


@torch.jit.script
def compute_imitation_reset_max(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    curr_rigid_body_pos: torch.Tensor,
    goal_rigid_body_pos: torch.Tensor,
    contact_body_ids: torch.Tensor,
    max_episode_length: float,
    enable_early_termination: bool,
    early_termination_dist_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        pos_dist = (
            (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=-1).sqrt()
        )  # (N,K)
        # print("pos_dist shape:", pos_dist.shape)
        # print("pos_dist:", pos_dist)
        if contact_body_ids.numel() > 0:
            pos_dist[:, contact_body_ids] = 0.0

        max_pos_dist = torch.max(pos_dist, dim=-1).values
        has_deviated = max_pos_dist > early_termination_dist_threshold
        # print("has_deviated:", has_deviated)
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1.0, torch.ones_like(reset_buf), terminated
    )
    return reset, terminated


@torch.jit.script
def compute_imitation_reset_max_posrot_upper(
    reset_buf: torch.Tensor,  # (N,)
    progress_buf: torch.Tensor,  # (N,)
    curr_rigid_body_pos: torch.Tensor,  # (N,K,3)  (rel to env origin)
    goal_rigid_body_pos: torch.Tensor,  # (N,K,3)
    curr_rigid_body_rot: torch.Tensor,  # (N,K,4)  (wxyz, normalized or close)
    goal_rigid_body_rot: torch.Tensor,  # (N,K,4)  (wxyz, normalized or close)
    contact_body_ids: torch.Tensor,  # (C,) local indices in K, can be empty
    upper_body_ids: torch.Tensor,  # (U,) local indices in K, non-empty recommended
    max_episode_length: float,
    enable_early_termination: bool,
    early_termination_dist_threshold: float,  # e.g. 0.2
    early_term_rot_threshold_deg: float,  # e.g. 90.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        # -------------------------
        # POS: reset-max (paper style)
        # -------------------------
        pos_dist = (
            (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=-1).sqrt()
        )  # (N,K)
        if contact_body_ids.numel() > 0:
            pos_dist[:, contact_body_ids] = 0.0
        max_pos_dist = torch.max(pos_dist, dim=-1).values  # (N,)
        has_pos_deviated = max_pos_dist > early_termination_dist_threshold

        # -------------------------
        # ROT: max angle error on upper body only
        # -------------------------
        curr_q = curr_rigid_body_rot
        goal_q = goal_rigid_body_rot

        # angular distance: angle = 2*acos(|dot|)
        dot = torch.sum(curr_q * goal_q, dim=-1).abs().clamp(0.0, 1.0)  # (N,K)
        ang = 2.0 * torch.acos(dot)  # rad, (N,K)
        rad2deg = 57.29577951308232  # 180/pi
        ang_deg = ang * rad2deg

        # focus on upper body only
        if upper_body_ids.numel() > 0:
            ang_deg_upper = ang_deg.index_select(1, upper_body_ids)  # (N,U)
            max_ang_deg = torch.max(ang_deg_upper, dim=-1).values  # (N,)
        else:
            # fallback: if empty, consider all (not recommended)
            max_ang_deg = torch.max(ang_deg, dim=-1).values

        has_rot_deviated = max_ang_deg > early_term_rot_threshold_deg

        # combine
        has_deviated = has_pos_deviated | has_rot_deviated
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1.0, torch.ones_like(reset_buf), terminated
    )
    return reset, terminated
