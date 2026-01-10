# g1_hybrid_gym/envs/g1_hybrid_gym_env_amp.py

from __future__ import annotations

import torch
from typing import Tuple

from .g1_hybrid_gym_env_base import G1HybridGymEnvBase
from g1_hybrid_prior.dataset_builder import make_dataset
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
        return make_dataset(cfg=cfg_params, device=self.device)

    def __init__(self, *args, **kwargs):
        # Chiamiamo il genitore. Questo scatenerà _build_reference_tensors.
        # Grazie al fix "lazy init" sotto, non crasherà più.
        super().__init__(*args, **kwargs)

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
        body_isaac_indices = []
        body_npz_indices = []

        # Tracciamo SOLO i body che esistono in entrambi (di solito saranno tutti)
        for isaac_name in isaac_body_names:
            if isaac_name in name_to_npz:
                body_isaac_indices.append(isaac_body_names.index(isaac_name))
                body_npz_indices.append(name_to_npz[isaac_name])

        if len(body_isaac_indices) == 0:
            raise ValueError(
                "[G1HybridGymEnvAMP] No overlapping body names between NPZ and Isaac."
            )

        self._body_isaac_indices = torch.tensor(
            body_isaac_indices, device=self.device, dtype=torch.long
        )
        self._body_npz_indices = torch.tensor(
            body_npz_indices, device=self.device, dtype=torch.long
        )

        # ------------- Contact bodies (da escludere nel maxdist) -------------
        # Tipicamente piedi
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

    def _build_reference_tensors(self):
        """Zero-copy: punta ai tensori del NPZ dataset + costruisce ref body pos per reset max."""
        if self.dataset is None:
            return

        # --- FIX: LAZY INITIALIZATION ---
        # Se siamo chiamati dal __init__ del padre, i mapping non esistono ancora.
        # Li creiamo ora.
        if not hasattr(self, "_body_npz_indices"):
            self._setup_body_mapping()
        # --------------------------------

        # Root (body 0 nel NPZ -> già estratto nel dataset)
        self._ref_root_pos = self.dataset.root_pos_w
        self._ref_root_quat_wxyz = self.dataset.root_rot_w

        root_rot = quat_normalize(self.dataset.root_rot_w)
        self._ref_root_lin_vel = quat_rotate_inv(root_rot, self.dataset.root_lin_vel_w)
        self._ref_root_ang_vel = quat_rotate_inv(root_rot, self.dataset.root_ang_vel_w)

        self._ref_joints = self.dataset.dof_pos
        self._ref_joint_vel = self.dataset.dof_vel

        # AMP: EE non serve
        self._ref_ee_pos = None

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

        # Reorder/select bodies to match tracked Isaac bodies
        self._ref_body_pos = body_pos_w.index_select(1, self._body_npz_indices)

    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()

        # amp_obs = s_cur = prima metà di (s_cur || goal)
        full_obs = obs_dict["policy"]
        half_len = full_obs.shape[-1] // 2
        s_cur = full_obs[..., :half_len]

        if self.extras is None:
            self.extras = {}
        self.extras["amp_obs"] = s_cur
        return obs_dict

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

        reset_f, term_f = compute_imitation_reset_max(
            reset_buf=reset_buf,
            progress_buf=progress_buf,
            curr_rigid_body_pos=curr_body_pos_rel,
            goal_rigid_body_pos=goal_body_pos,
            contact_body_ids=self._contact_body_ids,
            max_episode_length=float(self.max_episode_length),
            enable_early_termination=True,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        pos_dist = (
            (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=-1).sqrt()
        )  # (N,K)

        if contact_body_ids.numel() > 0:
            pos_dist[:, contact_body_ids] = 0.0

        max_pos_dist = torch.max(pos_dist, dim=-1).values
        has_deviated = max_pos_dist > 0.5
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1.0, torch.ones_like(reset_buf), terminated
    )
    return reset, terminated
