# g1_hybrid_gym/envs/g1_hybrid_gym_env_base.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import yaml
from typing import Dict, Optional
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
from g1_hybrid_prior.dataset_builder import make_dataset
from g1_hybrid_prior.helpers import (
    quat_normalize,
    quat_rotate,
    quat_rotate_inv,
    quat_mul,
    quat_inv,
    wrap_to_pi,
)

PARENT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()


class G1HybridGymEnvBase(DirectRLEnv):
    """Classe base comune tra PPO tracking e AMP.

    Responsabilità:
      - setup scena/robot
      - mapping giunti dataset <-> Isaac
      - action mapping (PD residual targets)
      - costruzione obs s_cur + goal (errore vs reference)
      - reset su frame random e write stato
      - reward tracking (DeepMimic-style) basato su ref "joints/joint_vel/root/ee (optional)"
    """

    cfg: G1HybridGymEnvCfg

    def __init__(
        self, cfg: G1HybridGymEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # ---------- Dataset ----------
        self.dataset = self._load_dataset()
        self.max_frame_idx = len(self.dataset) - 1
        print(
            f"[{self.__class__.__name__}] Loaded dataset with {len(self.dataset)} frames"
        )

        # ---------- Joint Mapping ----------
        dataset_joint_names = self.dataset.robot_cfg.joint_order
        isaac_joint_names = self.robot.joint_names

        dataset_to_isaac_indexes: list[int] = []
        g1_dof_idx: list[int] = []

        for name in dataset_joint_names:
            if name not in isaac_joint_names:
                raise ValueError(
                    f"[{self.__class__.__name__}] Joint '{name}' from dataset not found in robot articulation!"
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
        self._g1_dof_idx = torch.tensor(
            g1_dof_idx, device=self.device, dtype=torch.long
        )

        # sanity check (names)
        idx1 = self.dataset_to_isaac_indexes.tolist()
        idx2 = self._g1_dof_idx.tolist()
        names1 = [self.robot.joint_names[i] for i in idx1]
        names2 = [self.robot.joint_names[i] for i in idx2]
        if names1 != names2:
            raise ValueError(
                f"[{self.__class__.__name__}] Joint-name mismatch between dataset_to_isaac_indexes and _g1_dof_idx!\n"
                f"dataset_to_isaac: {names1}\n"
                f"g1_dof_idx:      {names2}"
            )

        # ---------- End-effector mapping (optional for derived envs) ----------
        self.ee_names = getattr(self.dataset.robot_cfg, "ee_link_names", [])
        self.ee_isaac_indices = None
        if self.ee_names:
            all_body_names = self.robot.body_names
            ee_isaac_indices = []
            for ee_name in self.ee_names:
                if ee_name not in all_body_names:
                    raise ValueError(
                        f"[{self.__class__.__name__}] End-Effector Link '{ee_name}' not found in Isaac articulation bodies!"
                    )
                ee_isaac_indices.append(all_body_names.index(ee_name))
            self.ee_isaac_indices = torch.tensor(
                ee_isaac_indices, device=self.device, dtype=torch.long
            )
            print(
                f"[{self.__class__.__name__}] Mapped EE: {self.ee_names} -> {self.ee_isaac_indices.tolist()}"
            )

        # ---------- PD action scaling ----------
        self._build_pd_action_offset_scale()

        # ---------- Reference indexing ----------
        self.ref_frame_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # cache ref per step (dones->rewards)
        self._cached_ref_tensors: Dict[str, torch.Tensor] | None = None

        # ref tensors (prestack / zero-copy depending on dataset)
        self._ref_root_pos: torch.Tensor | None = None
        self._ref_root_quat_wxyz: torch.Tensor | None = None
        self._ref_root_lin_vel: torch.Tensor | None = None
        self._ref_root_ang_vel: torch.Tensor | None = None
        self._ref_joints: torch.Tensor | None = None
        self._ref_joint_vel: torch.Tensor | None = None
        self._ref_ee_pos: torch.Tensor | None = None  # optional (PPO)
        self._ref_body_pos: torch.Tensor | None = None  # optional (AMP early term)
        self._ref_body_rot: torch.Tensor | None = None

        self._build_reference_tensors()

        # ---------- debug buffers ----------
        self._dbg_fallen: torch.Tensor | None = None
        self._dbg_ee_term: torch.Tensor | None = None
        self._dbg_maxdist: torch.Tensor | None = None
        self._dbg_action_abs_mean: torch.Tensor | None = None
        self._dbg_action_sat_frac: torch.Tensor | None = None

        self._dbg_joint_pos_mse: torch.Tensor | None = None
        self._dbg_joint_vel_mse: torch.Tensor | None = None
        self._dbg_root_pos_mse: torch.Tensor | None = None
        self._dbg_ee_pos_mse: torch.Tensor | None = None

    # --------------------------------------------------------------------- #
    # Dataset hooks
    # --------------------------------------------------------------------- #

    def _read_dataset_params(self) -> dict:
        cfg_params_path = str(PARENT_DIR / "config" / "config_param.yaml")
        with open(cfg_params_path, "r") as f:
            cfg_params = yaml.safe_load(f)["dataset_params"]
        return cfg_params

    def _load_dataset(self):
        cfg_params = self._read_dataset_params()
        return make_dataset(cfg=cfg_params, device=self.device)

    def _build_reference_tensors(self) -> None:
        """Override in children if needed (AMP uses zero-copy tensors from NPZ dataset)."""
        # default: prestack from self.dataset.dataset (list of dict)
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

        if "ee_pos" in frames[0]:
            self._ref_ee_pos = torch.stack([f["ee_pos"] for f in frames], dim=0).to(
                self.device
            )

    def _get_ref_batch(self, frame_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = frame_idx.clamp(0, self.max_frame_idx)
        batch = {
            "root_pos": self._ref_root_pos.index_select(0, idx),
            "root_quat_wxyz": self._ref_root_quat_wxyz.index_select(0, idx),
            "root_lin_vel": self._ref_root_lin_vel.index_select(0, idx),
            "root_ang_vel": self._ref_root_ang_vel.index_select(0, idx),
            "joints": self._ref_joints.index_select(0, idx),
            "joint_vel": self._ref_joint_vel.index_select(0, idx),
        }
        if self._ref_ee_pos is not None:
            batch["ee_pos"] = self._ref_ee_pos.index_select(0, idx)
        if self._ref_body_pos is not None:
            batch["body_pos"] = self._ref_body_pos.index_select(0, idx)
        if self._ref_body_rot is not None:
            batch["body_rot"] = self._ref_body_rot.index_select(0, idx)

        return batch

    # --------------------------------------------------------------------- #
    # Scene / actions
    # --------------------------------------------------------------------- #

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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

        pd_scale = 0.5 * (high - low)  # (J,)
        low_ext = mid - pd_scale
        high_ext = mid + pd_scale

        # self._pd_action_offset = mid.to(self.device)
        self._pd_action_scale = pd_scale.to(self.device)
        self._pd_action_limit_lower = low_ext.to(self.device)
        self._pd_action_limit_upper = high_ext.to(self.device)

    def _pre_physics_step(self, actions):
        if actions is None:
            self._target_q = None
            return

        a = actions.clamp(-1.0, 1.0).to(self.device)

        with torch.no_grad():
            self._dbg_action_abs_mean = a.abs().mean(dim=-1)
            self._dbg_action_sat_frac = (a.abs() > 0.95).float().mean(dim=-1)

        q0 = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]  # (N,J)
        delta = self._pd_action_scale.unsqueeze(0) * a
        target = q0 + delta
        target = torch.max(target, self._pd_action_limit_lower.unsqueeze(0))
        target = torch.min(target, self._pd_action_limit_upper.unsqueeze(0))
        self._target_q = target

    def _apply_action(self):
        if getattr(self, "_target_q", None) is None:
            return
        self.robot.set_joint_position_target(self._target_q, joint_ids=self._g1_dof_idx)

    # --------------------------------------------------------------------- #
    # Obs / goal
    # --------------------------------------------------------------------- #

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
        h_ref = ref["root_pos"][:, 2:3]
        dh = h_ref - h_cur

        q_ref = quat_normalize(ref["root_quat_wxyz"])
        q_cur = quat_normalize(q_cur_wxyz)
        q_err = quat_mul(q_ref, quat_inv(q_cur))
        q_err = quat_normalize(q_err)

        v_ref_body_ref = ref["root_lin_vel"]
        w_ref_body_ref = ref["root_ang_vel"]

        v_ref_world = quat_rotate(q_ref, v_ref_body_ref)
        w_ref_world = quat_rotate(q_ref, w_ref_body_ref)

        v_ref_body_sim = quat_rotate_inv(q_cur, v_ref_world)
        w_ref_body_sim = quat_rotate_inv(q_cur, w_ref_world)

        dv = v_ref_body_sim - v_cur_body
        dw = w_ref_body_sim - w_cur_body

        dq = wrap_to_pi(ref["joints"] - joint_pos)
        dqd = ref["joint_vel"] - joint_vel

        return torch.cat((dh, q_err, dv, dw, dq, dqd), dim=-1)

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

    # --------------------------------------------------------------------- #
    # Dones / reset / rewards
    # --------------------------------------------------------------------- #

    def _get_dones(self):
        """Base: solo timeout + fallen. I figli possono estendere con early termination."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        root_z_rel = (
            self.robot.data.root_link_state_w[:, 2] - self.scene.env_origins[:, 2]
        )
        fallen = root_z_rel < self.cfg.min_height_reset
        terminated = fallen

        # cache ref
        self.ref_frame_idx.clamp_(0, self.max_frame_idx)
        self._cached_ref_tensors = self._get_ref_batch(self.ref_frame_idx)

        self._dbg_fallen = fallen
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)
        n = len(env_ids)

        # RSI
        min_episode_frames = 60
        max_start = max(0, self.max_frame_idx - min_episode_frames)
        random_starts = torch.randint(0, max_start + 1, (n,), device=self.device)
        self.ref_frame_idx[env_ids] = random_starts

        # reference state
        root_pos_0 = self._ref_root_pos[random_starts]
        root_quat_0 = self._ref_root_quat_wxyz[random_starts]
        root_lin_vel_0_body = self._ref_root_lin_vel[random_starts]
        root_ang_vel_0_body = self._ref_root_ang_vel[random_starts]
        joints_0 = self._ref_joints[random_starts]
        joint_vel_0 = self._ref_joint_vel[random_starts]

        env_origins = self.scene.env_origins[env_ids]
        root_pos_w = root_pos_0 + env_origins
        root_quat_w = quat_normalize(root_quat_0)

        # body->world->COM velocity
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

        self._cached_ref_tensors = None

    def _get_rewards(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self.dataset_to_isaac_indexes]
        joint_vel = self.robot.data.joint_vel[:, self.dataset_to_isaac_indexes]

        root_link_state = self.robot.data.root_link_state_w
        root_pos_w = root_link_state[:, 0:3] - self.scene.env_origins
        root_quat_w = quat_normalize(root_link_state[:, 3:7])

        # EE (optional)
        ee_pos_rel = None
        if self.ee_isaac_indices is not None:
            ee_state_w = self.robot.data.body_state_w[:, self.ee_isaac_indices, 0:3]
            ee_pos_rel = ee_state_w - self.scene.env_origins.unsqueeze(1)

        # ref
        if self._cached_ref_tensors is not None:
            ref = self._cached_ref_tensors
        else:
            self.ref_frame_idx.clamp_(0, self.max_frame_idx)
            ref = self._get_ref_batch(self.ref_frame_idx)

        with torch.no_grad():
            self._dbg_joint_pos_mse = ((joint_pos - ref["joints"]) ** 2).mean(dim=-1)
            self._dbg_joint_vel_mse = ((joint_vel - ref["joint_vel"]) ** 2).mean(dim=-1)
            self._dbg_root_pos_mse = ((root_pos_w - ref["root_pos"]) ** 2).sum(dim=-1)

            if ref.get("ee_pos") is not None and ee_pos_rel is not None:
                self._dbg_ee_pos_mse = (
                    ((ee_pos_rel - ref["ee_pos"]) ** 2).sum(dim=-1).mean(dim=-1)
                )
            else:
                self._dbg_ee_pos_mse = None

        total_reward = compute_rewards(
            self.cfg.rew_w_pose,
            self.cfg.rew_w_vel,
            self.cfg.rew_w_root_pos,
            self.cfg.rew_w_root_rot,
            self.cfg.rew_w_ee,
            self.cfg.rew_alive,
            # current
            joint_pos,
            joint_vel,
            root_pos_w,
            root_quat_w,
            (
                ee_pos_rel
                if ee_pos_rel is not None
                else torch.zeros((self.num_envs, 1, 3), device=self.device)
            ),
            # ref
            ref["joints"],
            ref["joint_vel"],
            ref["root_pos"],
            ref["root_quat_wxyz"],
            ref.get("ee_pos", None),
            # flags
            self.reset_terminated,
        )

        alive = ~self.reset_buf
        self.ref_frame_idx[alive] += 1
        self.ref_frame_idx[self.ref_frame_idx > self.max_frame_idx] = 0

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

    joint_pos_err = torch.mean(torch.square(joint_pos - ref_joint_pos), dim=-1)
    joint_vel_err = torch.mean(torch.square(joint_vel - ref_joint_vel), dim=-1)
    root_pos_err = torch.sum(torch.square(root_pos - ref_root_pos), dim=-1)

    if ref_ee_pos is not None:
        ee_diff = ee_pos - ref_ee_pos
        ee_sq_err = torch.mean(torch.sum(torch.square(ee_diff), dim=-1), dim=-1)
        ee_total_err = ee_sq_err
    else:
        ee_total_err = torch.zeros_like(root_pos_err)

    r_pose = torch.exp(-rew_w_pose * joint_pos_err)
    r_vel = torch.exp(-rew_w_vel * joint_vel_err)
    r_root_p = torch.exp(-rew_w_root_pos * root_pos_err)
    r_ee = torch.exp(-rew_w_ee * ee_total_err)

    imitation_reward = r_pose * r_vel * r_root_p * r_ee
    return imitation_reward
