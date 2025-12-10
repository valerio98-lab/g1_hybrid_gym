from rl_games.algos_torch import model_builder
from .expert_low_level import ExpertLowLevelPolicy

model_builder.register_model("expert_low_level", ExpertLowLevelPolicy)
