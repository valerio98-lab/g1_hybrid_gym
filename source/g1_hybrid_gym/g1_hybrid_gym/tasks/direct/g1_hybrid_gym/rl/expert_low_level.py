import torch
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from g1_hybrid_prior.expert_policy import LowLevelExpertPolicy


class ExpertLowLevelPolicy(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        obs_shape = config["input_shape"]  # (127,)
        if len(obs_shape) != 1:
            raise RuntimeError(
                f"[ExpertLowLevelPolicy] Only flat obs supported, got shape={obs_shape}"
            )

        full_obs_dim = obs_shape[0]
        if (full_obs_dim) % 2 != 0:
            raise RuntimeError(
                f"[ExpertLowLevelPolicy] Expected obs = [s_cur, s_ref] s_cur(69) + s_ref(69) = 138, "
                f"but got dim={full_obs_dim}"
            )

        state_dim = (full_obs_dim) // 2
        obs_dim = state_dim  # [q, qdot]
        goal_dim = state_dim  # [q_ref - q, qdot_ref - qdot]
        action_dim = config["actions_num"]
        device = config.get("device", "cuda:0")

        print(
            f"[ExpertLowLevelPolicy] obs_dim={obs_dim}, goal_dim={goal_dim}, action_dim={action_dim}"
        )
        print(f"[ExpertLowLevelPolicy] CONFIG KEYS: {config.keys()}")

        expert_policy = LowLevelExpertPolicy(
            obs_dim=obs_dim,
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
            full_dim = self.obs_shape[0]

            if full_dim % 2 != 0:
                raise RuntimeError(
                    f"[ExpertLowLevelPolicy.Network] obs dim={full_dim} is not even; "
                    f"expected [s_cur, s_ref]"
                )
            self.state_dim = full_dim // 2  # dim di s_cur (e s_ref)

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)

            # obs_full = [s_cur, s_ref]
            obs_full = self.norm_obs(input_dict["obs"])
            state_dim = self.state_dim

            obs = obs_full[..., :state_dim]  # s_cur
            goal = obs_full[..., state_dim:]  # s_ref

            mu, log_std, value = self.a2c_network(obs, goal)
            sigma = torch.exp(log_std)

            if not hasattr(self, "_dbg_step"):
                self._dbg_step = 0
            self._dbg_step += 1

            if self._dbg_step % 200 == 0:
                with torch.no_grad():
                    # batch: (num_envs, action_dim)
                    mu0 = mu[0]
                    sig0 = sigma[0]
                    log0 = log_std[0]
                    print(
                        f"[pi_dbg] mu|mean={mu0.abs().mean().item():.3f} "
                        f"mu|p95={mu0.abs().quantile(0.95).item():.3f} "
                        f"sigma|mean={sig0.mean().item():.3f} "
                        f"sigma|minmax=({sig0.min().item():.3f},{sig0.max().item():.3f})"
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
                action = mu  # distr.sample()
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
