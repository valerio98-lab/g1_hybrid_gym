import torch

from .g1_hybrid_gym_env_base import G1HybridGymEnvBase


class G1HybridGymEnvPPO(G1HybridGymEnvBase):
    """Tracking PPO su dataset AUGMENTED (CSV pipeline)."""

    def _load_dataset(self):
        cfg_params = self._read_dataset_params()
        if cfg_params.get("training_type") == "ppo_amp":
            raise ValueError(
                "G1HybridGymEnvPPO doesn't support AMP. Use G1HybridGymEnvAMP."
            )
        return super()._load_dataset()

    def _get_dones(self):
        terminated, time_out = super()._get_dones()

        ref = self._cached_ref_tensors
        if ref is None:
            self.ref_frame_idx.clamp_(0, self.max_frame_idx)
            ref = self._get_ref_batch(self.ref_frame_idx)
            self._cached_ref_tensors = ref

        if ref.get("ee_pos") is not None and self.ee_isaac_indices is not None:
            ee_state_w = self.robot.data.body_state_w[:, self.ee_isaac_indices, 0:3]
            ee_pos_rel = ee_state_w - self.scene.env_origins.unsqueeze(1)
            max_dist = (
                torch.linalg.norm(ee_pos_rel - ref["ee_pos"], dim=-1).max(dim=-1).values
            )
            ee_term = max_dist > 0.5
            terminated = terminated | ee_term
            self._dbg_ee_term = ee_term
            self._dbg_maxdist = max_dist

        return terminated, time_out
