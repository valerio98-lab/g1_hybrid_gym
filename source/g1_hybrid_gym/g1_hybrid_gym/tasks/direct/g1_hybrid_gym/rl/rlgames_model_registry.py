from rl_games.algos_torch import model_builder
from .wrapper_ppo import ExpertPolicyWrapper

model_builder.register_model("wrapper_ppo", ExpertPolicyWrapper)
