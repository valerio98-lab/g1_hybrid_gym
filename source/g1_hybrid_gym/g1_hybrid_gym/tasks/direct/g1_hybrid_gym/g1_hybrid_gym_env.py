# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import Dict
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

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

        dataset_path = Path(
            "/home/valerio/g1_hybrid_prior/data_raw/LAFAN1_Retargeting_Dataset/g1_ee_augmented/walk1_subject1.csv"
        )
        self.dataset = G1HybridPriorDataset(
            file_path=dataset_path,
            robot="g1",
            dataset_type="augmented",
            lazy_load=False,
            vel_mode="central",
        )
        print(
            f"[G1HybridGymEnv] Loaded AUGMENTED dataset with {len(self.dataset)} frames"
        )

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
        )

        self._build_pd_action_offset_scale()

        idx1 = self.dataset_to_isaac_indexes.tolist()  # list[int]
        idx2 = self._g1_dof_idx.tolist()  # list[int]

        names1 = [self.robot.joint_names[i] for i in idx1]
        names2 = [self.robot.joint_names[i] for i in idx2]

        if names1 != names2:
            raise ValueError(
                "[G1HybridGymEnv] Joint-name mismatch between dataset_to_isaac_indexes and _g1_dof_idx!\n"
                f"dataset_to_isaac: {names1}\n"
                f"g1_dof_idx:      {names2}"
            )

        # --- End-Effector Mapping (Isaac Indices) ---
        # Cerchiamo gli indici dei body in Isaac che corrispondono agli EE del dataset
        self.ee_names = self.dataset.robot_cfg.ee_link_names
        self.ee_isaac_indices = []

        all_body_names = self.robot.body_names
        for ee_name in self.ee_names:
            if ee_name not in all_body_names:
                raise ValueError(
                    f"End-Effector Link '{ee_name}' not found in Isaac articulation bodies!"
                )
            self.ee_isaac_indices.append(all_body_names.index(ee_name))

        self.ee_isaac_indices = torch.tensor(
            self.ee_isaac_indices, device=self.device, dtype=torch.long
        )
        print(
            f"[G1HybridGymEnv] Mapped End-Effectors: {self.ee_names} -> Indices {self.ee_isaac_indices.tolist()}"
        )

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

        # -----------------------
        # TB debug buffers
        # -----------------------
        self._dbg_fallen: torch.Tensor | None = None
        self._dbg_ee_term: torch.Tensor | None = None
        self._dbg_maxdist: torch.Tensor | None = None

        self._dbg_joint_pos_mse: torch.Tensor | None = None
        self._dbg_joint_vel_mse: torch.Tensor | None = None
        self._dbg_root_pos_mse: torch.Tensor | None = None
        self._dbg_ee_pos_mse: torch.Tensor | None = None

        self._dbg_action_abs_mean: torch.Tensor | None = None
        self._dbg_action_sat_frac: torch.Tensor | None = None

    def _build_pd_action_offset_scale(self):
        lims0 = self.robot.data.default_joint_pos_limits[0]  # (J_all, 2)
        if not torch.isfinite(lims0).all():
            raise RuntimeError(
                "Joint limits not finite yet. Call _build_pd_action_offset_scale later."
            )

        idx = self.dataset_to_isaac_indexes  # (J,)
        low = lims0[idx, 0].clone()
        high = lims0[idx, 1].clone()

        mid = 0.5 * (high + low)

        # LORO: curr_scale = 0.7 * (curr_high - curr_low) e poi low/high = mid +/- curr_scale
        # => span esteso = 2*curr_scale, e quindi pd_action_scale = 0.5*span_esteso = curr_scale
        pd_scale = 0.5 * (
            high - low
        )  # (J,) #portatot a 0.5 per problemi di jittering e stabilità (0.5 = è esattamente la metà del range totale ossia il massimo fisicmanete possibile)
        low_ext = mid - pd_scale  # (J,)
        high_ext = mid + pd_scale  # (J,)

        self._pd_action_offset = mid.to(self.device)  # (J,) (qui coincide col mid)
        self._pd_action_scale = pd_scale.to(self.device)  # (J,)
        self._pd_action_limit_lower = low_ext.to(self.device)
        self._pd_action_limit_upper = high_ext.to(self.device)

        # ---- DEBUG: top joints by pd_scale ----
        vals, idxs = torch.topk(self._pd_action_scale.detach().abs().cpu(), k=8)
        names = [
            self.robot.joint_names[int(self.dataset_to_isaac_indexes[i].item())]
            for i in idxs
        ]
        print("[pd_scale_top] ", list(zip(names, vals.tolist())))

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
            self._ref_ee_pos = torch.stack([f["ee_pos"] for f in frames], dim=0).to(
                self.device
            )

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

        # --- DEBUG: dataset joint_vel consistency vs finite-diff of joints ---
        if self.common_step_counter % 500 == 0:
            with torch.no_grad():
                # prendi pochi env per non spammare
                ids = torch.arange(min(8, idx.shape[0]), device=idx.device)
                t = idx[ids]

                # clamp per evitare out-of-range
                t_prev = (t - 1).clamp(0, self.max_frame_idx)
                t_next = (t + 1).clamp(0, self.max_frame_idx)

                q_prev = self._ref_joints.index_select(0, t_prev)  # (K,J)
                q_next = self._ref_joints.index_select(0, t_next)  # (K,J)
                v_stored = self._ref_joint_vel.index_select(0, t)  # (K,J)

                # wrap per angular continuity
                dq_central = wrap_to_pi(q_next - q_prev)  # (K,J)

                dt_eff = float(
                    self.cfg.sim.dt * self.cfg.decimation
                )  # 0.03333... nel tuo caso
                v_fd_per_step = (
                    dq_central * 0.5
                )  # (K,J)  rad/step  (central diff senza /dt)
                v_fd_per_sec = v_fd_per_step / dt_eff  # (K,J)  rad/s

                # confronti di scala (ratio mediano)
                eps = 1e-8
                ratio_step = (
                    v_stored.abs().mean(dim=-1)
                    / (v_fd_per_step.abs().mean(dim=-1) + eps)
                ).median()
                ratio_sec = (
                    v_stored.abs().mean(dim=-1)
                    / (v_fd_per_sec.abs().mean(dim=-1) + eps)
                ).median()

                print(
                    f"[ds_vel_check] dt_eff={dt_eff:.6f}  "
                    f"|v_stored|mean={v_stored.abs().mean().item():.3f}  "
                    f"|v_fd_step|mean={v_fd_per_step.abs().mean().item():.3f}  "
                    f"|v_fd_sec|mean={v_fd_per_sec.abs().mean().item():.3f}  "
                    f"ratio(v/v_step)~{ratio_step.item():.3f}  ratio(v/v_sec)~{ratio_sec.item():.3f}"
                )
        # --- END DEBUG ---

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

    def _pre_physics_step(self, actions):
        if actions is None:
            self._target_q = None
            return

        a_no_clamp = actions.to(self.device)
        a = a_no_clamp.clamp(-1.0, 1.0)

        # --- TB: action stats per-env ---
        with torch.no_grad():
            self._dbg_action_abs_mean = a.abs().mean(dim=-1)  # (N,)
            self._dbg_action_sat_frac = (a.abs() > 0.95).float().mean(dim=-1)  # (N,)

        # current joint pos (serve per debug e per capire quanto "salti")
        q0 = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]  # (N,J)

        # absolute mapping (come paper)
        delta = self._pd_action_scale.unsqueeze(0) * a  # (N,J)
        target = q0 + delta  # residual: q_hat = q + a
        target = torch.max(target, self._pd_action_limit_lower.unsqueeze(0))
        target = torch.min(target, self._pd_action_limit_upper.unsqueeze(0))

        if self.common_step_counter % 500 == 0:
            with torch.no_grad():
                ref = self._get_ref_batch(
                    self.ref_frame_idx.clamp(0, self.max_frame_idx)
                )
                ref_q = ref["joints"]  # (N,J)

                dq_tgt = (target - q0).abs().mean().item()
                err_now = (q0 - ref_q).abs().mean().item()
                err_tgt = (target - ref_q).abs().mean().item()

                jv = self.robot.data.joint_vel[:, self.dataset_to_isaac_indexes]
                print(
                    f"[tgt_dbg] |tgt-q0|mean={dq_tgt:.3f}  |q0-ref|mean={err_now:.3f}  |tgt-ref|mean={err_tgt:.3f}  |jv|mean={jv.abs().mean().item():.3f}"
                )

                dq = (target - q0).abs().mean(dim=0)  # (J,)
                jv = (
                    self.robot.data.joint_vel[:, self.dataset_to_isaac_indexes]
                    .abs()
                    .mean(dim=0)
                )

                top = torch.topk(dq.cpu(), k=6)
                names = [
                    self.robot.joint_names[int(self.dataset_to_isaac_indexes[i].item())]
                    for i in top.indices
                ]
                print("[dq_top] ", list(zip(names, top.values.tolist())))

                topv = torch.topk(jv.cpu(), k=6)
                namesv = [
                    self.robot.joint_names[int(self.dataset_to_isaac_indexes[i].item())]
                    for i in topv.indices
                ]
                print("[jv_top] ", list(zip(namesv, topv.values.tolist())))

        self._target_q = target

        # --- DEBUG ACTIONS (every N steps) ---
        if self.common_step_counter % 200 == 0:
            a_pre_clamp = a_no_clamp
            a_print = a
            sa = self._pd_action_scale.unsqueeze(0) * a
            sat = (a_print.abs() > 0.95).float().mean().item()
            sat_pre = (a_pre_clamp.abs() > 0.95).float().mean().item()
            mean_abs = a_print.abs().mean().item()
            mean_abs_pre = a_pre_clamp.abs().mean().item()
            sa_mean_abs = sa.abs().mean().item()
            k = max(1, int(0.95 * a_print.numel()))
            k_pre = max(1, int(0.95 * a_pre_clamp.numel()))
            p95 = a_print.abs().flatten().kthvalue(k).values.item()
            p95_pre = a_pre_clamp.abs().flatten().kthvalue(k_pre).values.item()
            print(
                f"[actions post clamp] mean|a|={mean_abs:.3f}  sat%={sat*100:.1f}%  approx_p95|a|={p95:.3f}"
            )
            print(
                f"[actions pre clamp] mean|a|={mean_abs_pre:.3f} sat%={sat_pre*100:.1f}%  approx_p95|a|={p95_pre:.3f}"
            )
            print(
                f"[scaled_actions] mean|a|={mean_abs:.3f} sat%={sat*100:.1f}%  mean|scaled|={sa_mean_abs:.3f}"
            )
            print(
                "pd_scale min/max:",
                self._pd_action_scale.min().item(),
                self._pd_action_scale.max().item(),
            )
            print(
                "low/high ext min/max:",
                self._pd_action_limit_lower.min().item(),
                self._pd_action_limit_upper.max().item(),
            )
        # --- DEBUG ACTIONS (every N steps) ---

    def _apply_action(self):
        if getattr(self, "_target_q", None) is None:
            return
        self.robot.set_joint_position_target(self._target_q, joint_ids=self._g1_dof_idx)

    def step(self, action):
        """
        This step function extends the step function of DirectRLEnv by adding
        logging of various debug statistics when an episode ends. In a nutshell,
        It's a wrapper around the parent step() that just adds logging on done episodes,
        basically is useless from a functionality/physics/simulation point of view.
        """
        obs, rew, terminated, truncated, extras = super().step(action)

        done = terminated | truncated
        if not done.any():
            return obs, rew, terminated, truncated, extras

        if extras is None:
            extras = {}
        log = extras.setdefault("log", {})

        ids = done.nonzero(as_tuple=False).squeeze(-1)

        def _mean(x: torch.Tensor) -> torch.Tensor:
            return x.index_select(0, ids).mean()

        def _p95(x: torch.Tensor) -> torch.Tensor:
            v = x.index_select(0, ids).flatten()
            k = max(1, int(0.95 * v.numel()))
            return v.kthvalue(k).values

        def _max(x: torch.Tensor) -> torch.Tensor:
            v = x.index_select(0, ids)
            return v.max()

        # ---- dones ----
        if self._dbg_fallen is not None:
            log["fallen_pct"] = (_mean(self._dbg_fallen.float()) * 100.0).item()

        if self._dbg_ee_term is not None:
            log["ee_term_pct"] = (_mean(self._dbg_ee_term.float()) * 100.0).item()

        if self._dbg_maxdist is not None:
            log["ee_maxdist_mean"] = float(_mean(self._dbg_maxdist).item())
            log["ee_maxdist_p95"] = float(_p95(self._dbg_maxdist).item())
            log["ee_maxdist_max"] = float(_max(self._dbg_maxdist).item())

        # ---- MSE terms ----
        if self._dbg_joint_pos_mse is not None:
            log["mse_joint_pos"] = _mean(self._dbg_joint_pos_mse).item()
        if self._dbg_joint_vel_mse is not None:
            log["mse_joint_vel"] = _mean(self._dbg_joint_vel_mse).item()
        if self._dbg_root_pos_mse is not None:
            log["mse_root_pos"] = _mean(self._dbg_root_pos_mse).item()
        if self._dbg_ee_pos_mse is not None:
            log["mse_ee_pos"] = _mean(self._dbg_ee_pos_mse).item()

        # ---- action stats (arrivate all'env) ----
        if self._dbg_action_abs_mean is not None:
            log["action_abs_mean"] = _mean(self._dbg_action_abs_mean)
        if self._dbg_action_sat_frac is not None:
            log["action_sat_pct"] = _mean(self._dbg_action_sat_frac) * 100.0

        return obs, rew, terminated, truncated, extras

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

        self.ref_frame_idx.clamp_(0, self.max_frame_idx)
        ref = self._get_ref_batch(self.ref_frame_idx)

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

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        root_z_rel = (
            self.robot.data.root_link_state_w[:, 2] - self.scene.env_origins[:, 2]
        )
        fallen = root_z_rel < self.cfg.min_height_reset
        terminated = fallen

        self.ref_frame_idx.clamp_(0, self.max_frame_idx)
        ref = self._get_ref_batch(self.ref_frame_idx)
        self._cached_ref_tensors = ref

        # --- TB buffers ---
        self._dbg_fallen = fallen

        if ref.get("ee_pos") is not None:
            ee_state_w = self.robot.data.body_state_w[:, self.ee_isaac_indices, 0:3]
            ee_pos_rel = ee_state_w - self.scene.env_origins.unsqueeze(1)
            max_dist = (
                torch.linalg.norm(ee_pos_rel - ref["ee_pos"], dim=-1).max(dim=-1).values
            )

            ee_term = max_dist > 0.5
            terminated = terminated | ee_term

            # --- TB buffers ---
            self._dbg_maxdist = max_dist
            self._dbg_ee_term = ee_term

            if self.common_step_counter % 200 == 0:
                with torch.no_grad():
                    fallen_pct = fallen.float().mean().item() * 100
                    ee_pct = ee_term.float().mean().item() * 100
                    md_mean = max_dist.mean().item()
                    md_p95 = max_dist.quantile(0.95).item()
                    md_max = max_dist.max().item()
                    print(
                        f"[done_dbg] fallen%={fallen_pct:.1f} ee_term%={ee_pct:.1f} "
                        f"maxdist mean={md_mean:.3f} p95={md_p95:.3f} max={md_max:.3f}"
                    )
        else:
            # --- TB buffers ---
            self._dbg_maxdist = None
            self._dbg_ee_term = None

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
        self.robot.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, None, env_ids
        )

        # # --- DEBUG: check EE alignment right after reset ---
        # # (run only once, and only if env 0 was among the reset ids)
        # if (env_ids == 0).any() and not getattr(self, "_dbg_ee_checked", False):
        #     self._dbg_ee_checked = True

        #     # refresh kinematics without advancing dynamics
        #     self.scene.write_data_to_sim()
        #     self.sim.forward()
        #     self.scene.update(dt=0.0)

        #     # evaluate ONLY on the envs that were just reset (env_ids)
        #     ids = env_ids

        #     # ref for those ids
        #     ref = self._get_ref_batch(self.ref_frame_idx[ids].clamp(0, self.max_frame_idx))

        #     if ref.get("ee_pos") is not None:
        #         ee_state_w = self.robot.data.body_state_w[ids][:, self.ee_isaac_indices, 0:3]
        #         ee_pos_rel = ee_state_w - self.scene.env_origins[ids].unsqueeze(1)

        #         max_dist = torch.linalg.norm(ee_pos_rel - ref["ee_pos"], dim=-1).max(dim=-1).values
        #         print(f"[DEBUG reset] ee max_dist (reset envs only): mean={max_dist.mean().item():.3f}  max={max_dist.max().item():.3f}")

        self._cached_ref_tensors = None

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

        # --- TB: error terms per-env (MSE) ---
        with torch.no_grad():
            self._dbg_joint_pos_mse = ((joint_pos - ref["joints"]) ** 2).mean(
                dim=-1
            )  # (N,)
            self._dbg_joint_vel_mse = ((joint_vel - ref["joint_vel"]) ** 2).mean(
                dim=-1
            )  # (N,)
            self._dbg_root_pos_mse = ((root_pos_w - ref["root_pos"]) ** 2).sum(
                dim=-1
            )  # (N,)

            if ref.get("ee_pos") is not None:
                self._dbg_ee_pos_mse = (
                    ((ee_pos_rel - ref["ee_pos"]) ** 2).sum(dim=-1).mean(dim=-1)
                )  # (N,)
            else:
                self._dbg_ee_pos_mse = None

        # --- DEBUG REWARD TERMS (every N steps) ---
        if self.common_step_counter % 200 == 0:
            with torch.no_grad():
                joint_pos_err = torch.mean((joint_pos - ref["joints"]) ** 2, dim=-1)
                joint_vel_err = torch.mean((joint_vel - ref["joint_vel"]) ** 2, dim=-1)

                root_pos_err = torch.sum((root_pos_w - ref["root_pos"]) ** 2, dim=-1)

                # quat_dot = torch.sum(root_quat_w * ref["root_quat_wxyz"], dim=-1).abs().clamp(0.0, 1.0)
                # root_rot_err = 1.0 - quat_dot

                if ref.get("ee_pos") is not None:
                    ee_sq_err = torch.sum(
                        (ee_pos_rel - ref["ee_pos"]) ** 2, dim=-1
                    )  # (N,4)
                    ee_total_err = torch.mean(ee_sq_err, dim=-1)  # (N,)
                else:
                    ee_total_err = torch.zeros_like(root_pos_err)

                r_pose = torch.exp(-self.cfg.rew_w_pose * joint_pos_err)
                r_vel = torch.exp(-self.cfg.rew_w_vel * joint_vel_err)
                r_root_p = torch.exp(-self.cfg.rew_w_root_pos * root_pos_err)
                # r_root_r = torch.exp(-self.cfg.rew_w_root_rot * root_rot_err)
                r_ee = torch.exp(-self.cfg.rew_w_ee * ee_total_err)

                print(
                    f"[rew_terms] "
                    f"r_pose={r_pose.mean().item():.2e} "
                    f"r_vel={r_vel.mean().item():.2e} "
                    f"r_root_p={r_root_p.mean().item():.2e} "
                    f"r_ee={r_ee.mean().item():.2e} "
                    f"| ee_err={ee_total_err.mean().item():.2e} root_err={root_pos_err.mean().item():.2e}"
                )

                # --- DEBUG joint vel (ogni 500 step, solo env 0) ---
        if self.common_step_counter % 500 == 0:
            e = 0  # env index
            jv = joint_vel[e]  # (29,)
            rjv = ref["joint_vel"][e]  # (29,)
            diff = jv - rjv
            w = self.cfg.rew_w_vel
            mse_sum = diff.pow(2).sum()
            mse_mean = diff.pow(2).mean()
            print(
                f"[jvel_dbg] exp(-w*sum)={torch.exp(-w*mse_sum).item():.3e}  exp(-w*mean)={torch.exp(-w*mse_mean).item():.3e}"
            )

            print(
                "[jvel_dbg] "
                f"|jv|mean={jv.abs().mean().item():.3f}  "
                f"|ref|mean={rjv.abs().mean().item():.3f}  "
                f"diff|mean={diff.abs().mean().item():.3f}  "
                f"diff|p95={diff.abs().quantile(0.95).item():.3f}  "
                f"MSE_mean={(diff.pow(2).mean()).item():.3e}  "
                f"MSE_sum={(diff.pow(2).sum()).item():.3e}"
            )

        total_reward = compute_rewards(
            self.cfg.rew_w_pose,
            self.cfg.rew_w_vel,
            self.cfg.rew_w_root_pos,
            self.cfg.rew_w_root_rot,
            self.cfg.rew_w_ee,
            self.cfg.rew_alive,
            # Current Robot State
            joint_pos,
            joint_vel,
            root_pos_w,
            root_quat_w,
            ee_pos_rel,
            # Reference State
            ref["joints"],
            ref["joint_vel"],
            ref["root_pos"],
            ref["root_quat_wxyz"],
            ref.get("ee_pos"),
            # Flags
            self.reset_terminated,
        )

        alive = ~self.reset_buf
        self.ref_frame_idx[alive] += 1
        self.ref_frame_idx[self.ref_frame_idx > self.max_frame_idx] = 0

        # IMPORTANT: clear cache (so next step dones recomputes)
        self._cached_ref_tensors = None
        return total_reward


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
    ref_ee_pos: Optional[torch.Tensor],
    # Flags
    reset_terminated: torch.Tensor,
) -> torch.Tensor:

    # DeepMimic Style Rewards
    joint_pos_err = torch.mean(torch.square(joint_pos - ref_joint_pos), dim=-1)
    # Velocity joints
    joint_vel_err = torch.mean(torch.square(joint_vel - ref_joint_vel), dim=-1)
    # Root position
    root_pos_err = torch.sum(torch.square(root_pos - ref_root_pos), dim=-1)
    # Root rotation (1 - |dot|)
    # quat_dot = torch.sum(root_quat * ref_root_quat, dim=-1).abs().clamp(0.0, 1.0)
    # root_rot_err = 1.0 - quat_dot #(Errore su rotazione, non necessario al momento)

    # End-Effector Error
    if ref_ee_pos is not None:
        ee_diff = ee_pos - ref_ee_pos
        # Errore quadro per ogni EE, sommato su XYZ -> (N, 4)
        ee_sq_err = torch.mean(torch.sum(torch.square(ee_diff), dim=-1), dim=-1)
        ee_total_err = ee_sq_err
    else:
        ee_total_err = torch.zeros_like(root_pos_err)

    # Kernel Esponenziali (Moltiplicativi) sempre DeepMimic Style

    r_pose = torch.exp(-rew_w_pose * joint_pos_err)
    r_vel = torch.exp(-rew_w_vel * joint_vel_err)
    r_root_p = torch.exp(-rew_w_root_pos * root_pos_err)
    # r_root_r = torch.exp(-rew_w_root_rot * root_rot_err)
    r_ee = torch.exp(-rew_w_ee * ee_total_err)

    # 3. Reward Totale
    # Struttura moltiplicativa per l'imitazione: forza il rispetto di tutti i vincoli
    imitation_reward = r_pose * r_vel * r_root_p * r_ee

    # Bonus sopravvivenza additivo (per gradienti stabili all'inizio)
    # alive_bonus = rew_alive * (1.0 - reset_terminated.float())

    return imitation_reward
