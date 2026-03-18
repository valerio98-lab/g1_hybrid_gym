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
        self._bgd += 1

        ref = self._cached_ref_tensors
        if ref is None:
            self.ref_frame_idx.clamp_(0, self.max_frame_idx)
            ref = self._get_ref_batch(self.ref_frame_idx)
            self._cached_ref_tensors = ref

        if ref.get("body_pos") is not None and self.body_isaac_indices is not None:
            root_pos_w = self.robot.data.root_link_state_w[:, :3]
            body_state_w = self.robot.data.body_state_w[:, self.body_isaac_indices, 0:3]
            sim_pos_rel = body_state_w - root_pos_w.unsqueeze(1)  # relativo al root

            # ref["body_pos"] è già relativo al root → confronto diretto
            max_dist = torch.linalg.norm(sim_pos_rel - ref["body_pos"], dim=-1).max(dim=-1).values
            # max_dist_b = (
            #     torch.linalg.norm(ee_pos_rel[:, 4:, :] - ref["body_pos"][:, torch.tensor([6,7]), :], dim=-1).max(dim=-1).values
            # )
            # if self._bgd % 100 == 0:
            #     res = max_dist[:]
            #     print(f"Error values related to hands: {res}", flush=True)
            #     print(f"Error values in max_dist_b {max_dist_b}", flush=True)
            body_term = max_dist > 0.2
            terminated = terminated | body_term
            self._dbg_ee_term = body_term
            self._dbg_maxdist = max_dist

        return terminated, time_out
