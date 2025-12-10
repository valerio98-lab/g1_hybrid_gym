import torch
import yaml
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from g1_hybrid_prior.expert_policy import LowLevelExpertPolicy


class ExpertLowLevelPolicy(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        """
        RL-Games use this method to build the policy network.
        """
        obs_shape = config["input_shape"]  #(116,)
        if len(obs_shape) != 1:
            raise RuntimeError(
                f"[ExpertLowLevelPolicy] Only flat obs supported, got shape={obs_shape}"
            )

        full_obs_dim = obs_shape[0]  # 4 * n_joints expected
        if full_obs_dim % 4 != 0:
            raise RuntimeError(
                f"[ExpertLowLevelPolicy] Expected obs = [q, qdot, q_ref, qdot_ref] with dim 4 * n_joints, "
                f"but got dim={full_obs_dim}"
            )

        n_joints = full_obs_dim // 4
        obs_dim = 2 * n_joints  # [q, qdot]
        goal_dim = 2 * n_joints  # [q_ref - q, qdot_ref - qdot]
        action_dim = config["actions_num"]
        device = config.get("device", "cuda:0")

        print(
            f"[ExpertLowLevelPolicy] full_obs_dim={full_obs_dim}, n_joints={n_joints}"
        )
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
            n_joints=n_joints,
        )

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, n_joints, **kwargs):
            super().__init__(a2c_network, **kwargs)
            self.n_joints = n_joints

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)

            obs = self.norm_obs(input_dict["obs"])
            joint_pos = obs[:, 0 : self.n_joints]
            joint_vel = obs[:, self.n_joints : 2 * self.n_joints]
            ref_pos = obs[:, 2 * self.n_joints : 3 * self.n_joints]
            ref_vel = obs[:, 3 * self.n_joints : 4 * self.n_joints]

            obs_final_state = torch.cat([joint_pos, joint_vel], dim=-1)

            # goal = errore rispetto al mocap
            goal = torch.cat(
                [ref_pos - joint_pos, ref_vel - joint_vel],
                dim=-1,
            )

            mu, log_std, value = self.a2c_network(obs_final_state, goal)
            value = value.squeeze(-1)
            sigma = torch.exp(log_std)

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
                    "neglopacs": torch.squeeze(neglogp),
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
