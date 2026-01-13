# g1_hybrid_gym/skrl_utils.py

import torch.nn as nn
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model


# =============================================================================
# WRAPPER POLICY (ACTOR) - Nessuna modifica qui
# =============================================================================
class PolicyWrapperAMP(GaussianMixin, Model):
    def __init__(
        self, observation_space, action_space, device, actor_net, clip_actions=False
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
        )
        self.actor = actor_net

    def compute(self, inputs, role=""):
        x = inputs["states"]
        half_dim = x.shape[-1] // 2
        obs = x[..., :half_dim]
        target = x[..., half_dim:]

        mu, log_std = self.actor(obs, target)
        return mu, log_std, {}


# =============================================================================
# WRAPPER VALUE (CRITIC) - Nessuna modifica qui
# =============================================================================
class PolicyValueWrapper(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, critic_net):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.critic = critic_net

    def compute(self, inputs, role=""):
        x = inputs["states"]
        half_dim = x.shape[-1] // 2
        obs = x[..., :half_dim]
        target = x[..., half_dim:]

        value = self.critic(obs, target)
        return value, {}


# =============================================================================
# DISCRIMINATORE AMP - FIX LOGICA DI TAGLIO
# =============================================================================
class AMPDiscriminator(DeterministicMixin, Model):
    def __init__(
        self, observation_space, action_space, device, input_dim=None, per_step_dim=69
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        if input_dim is None:
            raise ValueError("AMPDiscriminator deve ricevere esplicitamente input_dim!")

        self.input_dim = int(input_dim)  # es: 138
        self.per_step_dim = int(per_step_dim)  # es: 69
        assert self.input_dim % self.per_step_dim == 0
        self.K = self.input_dim // self.per_step_dim

        print(
            f"[AMPDiscriminator] Initialized with Input Dim: {self.input_dim} (K={self.K})"
        )

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self._dbg_counter = 0

    def compute(self, inputs, role=""):
        x = inputs["states"]

        # 1) HARD ASSERT: dimensione deve essere ESATTAMENTE quella prevista
        if x.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"[AMPDiscriminator] Expected {self.input_dim}, got {x.shape[-1]} "
                f"(non tronco: voglio vedere il bug)"
            )

        return self.net(x), {}
