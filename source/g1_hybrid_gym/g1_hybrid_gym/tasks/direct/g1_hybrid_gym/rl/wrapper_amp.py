# g1_hybrid_gym/skrl_utils.py

import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model


# =============================================================================
# WRAPPER POLICY (ACTOR) - Nessuna modifica qui
# =============================================================================
class PolicyWrapperAMP(GaussianMixin, Model):
    def __init__(
        self, observation_space, action_space, device, expert_net, clip_actions=False
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
        )
        self.net = expert_net

    def compute(self, inputs, role=""):
        x = inputs["states"]
        half_dim = x.shape[-1] // 2
        obs = x[..., :half_dim]
        target = x[..., half_dim:]
        mu, log_std, _ = self.net(obs, target)
        return mu, log_std, {}


# =============================================================================
# WRAPPER VALUE (CRITIC) - Nessuna modifica qui
# =============================================================================
class PolicyValueWrapper(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, expert_net):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = expert_net

    def compute(self, inputs, role=""):
        x = inputs["states"]
        half_dim = x.shape[-1] // 2
        obs = x[..., :half_dim]
        target = x[..., half_dim:]
        _, _, value = self.net(obs, target)
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
            self.input_dim = observation_space.shape[0] // 2
        else:
            self.input_dim = input_dim

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

        # Altrimenti, se è doppio (138), taglialo.
        # Questo potrebbe succedere se SKRL passasse l'obs intera per sbaglio.
        elif x.shape[-1] == self.input_dim * 2:
            amp_obs = x[..., : self.input_dim]
            return self.net(amp_obs), {}

        # Fallback di sicurezza (o errore se dimensioni strane)
        else:
            # Proviamo a tagliare a metà come default, ma stampiamo un warning se serve debug
            half = x.shape[-1] // 2
            return self.net(x[..., :half]), {}
