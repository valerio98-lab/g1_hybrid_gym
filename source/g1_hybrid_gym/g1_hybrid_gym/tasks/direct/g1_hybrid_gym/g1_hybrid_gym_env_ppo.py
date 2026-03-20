import torch
import einops

from .g1_hybrid_gym_env_base import G1HybridGymEnvBase

from g1_hybrid_prior.helpers import quat_normalize, quat_rotate_inv


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
        self._bgd += 1

        ref = self._cached_ref_tensors
        if ref is None:
            self.ref_frame_idx.clamp_(0, self.max_frame_idx)
            ref = self._get_ref_batch(self.ref_frame_idx)
            self._cached_ref_tensors = ref

        if ref.get("body_pos") is not None and self.body_isaac_indices is not None:
            N = self.robot.data.root_link_state_w.shape[0]
            K = self.body_isaac_indices.shape[0]
            
            root_pos_w = self.robot.data.root_link_state_w[:, :3]
            sim_quat_w = quat_normalize(self.robot.data.root_link_state_w[:, 3:7])
            body_state_w = self.robot.data.body_state_w[:, self.body_isaac_indices, 0:3]
            
            sim_pos_rel_w = body_state_w - root_pos_w.unsqueeze(1) # Differenza in World
            sim_q_exp = einops.repeat(sim_quat_w, 'N dim -> N K dim', K=K)
            sim_pos_local = quat_rotate_inv(sim_q_exp, sim_pos_rel_w)

            ref_pos_rel_w = ref["body_pos"]
            ref_quat_w = quat_normalize(ref["root_quat_wxyz"])
            
            ref_q_exp = einops.repeat(ref_quat_w, 'N dim -> N K dim', K=K)
            ref_pos_local = quat_rotate_inv(ref_q_exp, ref_pos_rel_w)

            max_dist = torch.linalg.norm(sim_pos_local - ref_pos_local, dim=-1).max(dim=-1).values
            
            body_term = max_dist > 0.3
            terminated = terminated | body_term
            self._dbg_ee_term = body_term
            self._dbg_maxdist = max_dist

        return terminated, time_out
