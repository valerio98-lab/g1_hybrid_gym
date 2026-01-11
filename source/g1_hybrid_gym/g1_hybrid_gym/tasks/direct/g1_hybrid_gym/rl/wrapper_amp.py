# g1_hybrid_gym/skrl_utils.py

import torch
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
    def __init__(self, observation_space, action_space, device, input_dim=None):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # Calcoliamo la dimensione attesa (69)
        if input_dim is None:
            raise ValueError("AMPDiscriminator deve ricevere esplicitamente input_dim!")

        self.input_dim = input_dim
        print(f"[AMPDiscriminator] Initialized with Input Dim: {self.input_dim}")

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def compute(self, inputs, role=""):
        x = inputs["states"]

        # --- FIX INTELLIGENTE ---
        # Se l'input ha già la dimensione giusta (69), usalo così com'è.
        # Questo succede durante il training update (dati presi dalla memoria AMP).
        if x.shape[-1] == self.input_dim:
            return self.net(x), {}

        # Safety check opzionale
        if x.shape[-1] != self.input_dim:
            # Caso limite: se per qualche motivo arrivasse qualcosa di strano
            # proviamo a tagliare solo se è ESATTAMENTE il doppio ma stampiamo warning
            if x.shape[-1] == self.input_dim * 2:
                print(
                    f"[AMPDiscriminator] ⚠️ Warning: Input dimension {x.shape[-1]} is double the expected {self.input_dim}. Truncating input."
                )
                return self.net(x[..., : self.input_dim]), {}

            # Altrimenti crasha per avvisarci del mismatch
            raise RuntimeError(
                f"Discriminator expected {self.input_dim}, got {x.shape[-1]}"
            )

        return self.net(x), {}
