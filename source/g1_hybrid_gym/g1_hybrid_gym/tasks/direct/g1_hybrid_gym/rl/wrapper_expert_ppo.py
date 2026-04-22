import torch
import yaml
from functools import lru_cache
from pathlib import Path

from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from g1_hybrid_prior.models.expert_policy import ExpertPolicy

_CONFIG_PARAM_PATH = Path(__file__).resolve().parents[5] / "config" / "config_param.yaml"


@lru_cache(maxsize=1)
def _compute_obs_dims() -> tuple[int, int]:
    """Compute (cur_obs_dim, goal_dim) from robots.yaml — no motion data loaded."""
    from g1_hybrid_prior.dataset.robot_cfg import load_robot_cfg
    from g1_hybrid_prior.helpers import get_project_root

    params = yaml.safe_load(_CONFIG_PARAM_PATH.read_text())["dataset_params"]
    robots_yaml = str(get_project_root() / "config" / "robots.yaml")
    robot_cfg = load_robot_cfg(robots_yaml, params["robot"])

    J = robot_cfg.dof
    num_ee = robot_cfg.num_ee
    K = robot_cfg.num_body

    cur_obs_dim = 1 + 4 + 3 + 3 + J + J + num_ee * 3
    goal_dim    = 1 + 4 + 3 + 3 + J + J + K * 13
    print(
        f"[wrapper_expert_ppo] dims from robots.yaml: "
        f"J={J}, num_ee={num_ee}, K={K} → cur_obs_dim={cur_obs_dim}, goal_dim={goal_dim}"
    )
    return cur_obs_dim, goal_dim


class ExpertPolicyWrapper(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        obs_shape = config["input_shape"]
        if len(obs_shape) != 1:
            raise RuntimeError(
                f"[ExpertPolicyWrapper] Only flat obs supported, got shape={obs_shape}"
            )

        cur_obs_dim, goal_dim = _compute_obs_dims()

        full_obs_dim = obs_shape[0]
        assert full_obs_dim == cur_obs_dim + goal_dim, (
            f"[ExpertPolicyWrapper] obs mismatch: input_shape={full_obs_dim}, "
            f"expected {cur_obs_dim + goal_dim} (cur_obs={cur_obs_dim}, goal={goal_dim})"
        )

        action_dim = config["actions_num"]
        device = config.get("device", "cuda:0")

        print(
            f"[ExpertPolicyWrapper] obs_dim={cur_obs_dim}, goal_dim={goal_dim}, action_dim={action_dim}"
        )
        print(f"[ExpertPolicyWrapper] CONFIG KEYS: {config.keys()}", flush=True)

        expert_policy = ExpertPolicy(
            obs_dim=cur_obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            device=device,
        )

        value_size = config.get("value_size", 1)
        normalize_value = config["normalize_value"]
        normalize_input = config["normalize_input"]

        return self.Network(
            expert_policy,
            obs_shape=obs_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            value_size=value_size,
        )

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            self.obs_dim, _ = _compute_obs_dims()

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)

            obs_full = self.norm_obs(input_dict["obs"])
            obs  = obs_full[..., : self.obs_dim]
            goal = obs_full[..., self.obs_dim :]
            # print(
            #     f"[ExpertPolicyWrapper]: OBS_DIM: {obs.shape[-1]}, GOAL_DIM: {goal.shape[-1]}"
            # )

            mu, log_std, value = self.a2c_network(obs, goal)
            sigma = torch.exp(log_std)

            if not hasattr(self, "_dbg_step"):
                self._dbg_step = 0
            self._dbg_step += 1

            if self._dbg_step % 200 == 0:
                with torch.no_grad():
                    mu0  = mu[0]
                    sig0 = sigma[0]
                    log0 = log_std[0]
                    print(
                        f"[pi_dbg] mu|mean={mu0.abs().mean().item():.3f} "
                        f"mu|p95={mu0.abs().quantile(0.95).item():.3f} "
                        f"sigma|mean={sig0.mean().item():.3f} "
                        f"sigma|minmax=({sig0.min().item():.3f},{sig0.max().item():.3f}) "
                        f"log_std|mean={log0.mean().item():.3f} "
                        f"log_std|minmax=({log0.min().item():.3f},{log0.max().item():.3f})"
                    )

            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                if prev_actions is None:
                    raise RuntimeError("prev_actions must be provided during training")
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, log_std)
                return {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": None,
                    "mus": mu,
                    "sigmas": sigma,
                }
            else:
                action = distr.sample()
                neglogp = self.neglogp(action, mu, sigma, log_std)
                return {
                    "neglogpacs": torch.squeeze(neglogp),
                    "values": self.denorm_value(value),
                    "actions": action,
                    "rnn_states": None,
                    "mus": mu,
                    "sigmas": sigma,
                }

        def get_aux_loss(self):
            return None

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def get_value_layer(self):
            return None
