from rl_games.algos_torch import model_builder
from .wrapper_expert_ppo import ExpertPolicyWrapper

model_builder.register_model("wrapper_expert_ppo", ExpertPolicyWrapper)
