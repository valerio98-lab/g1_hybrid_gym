from rl_games.algos_torch import model_builder
from .wrapper_expert_ppo import ExpertPolicyWrapper
from .wrapper_task_ppo import TaskPolicyWrapper

model_builder.register_model("wrapper_expert_ppo", ExpertPolicyWrapper)

model_builder.register_model("wrapper_expert_task_ppo", TaskPolicyWrapper)

class _DummyNetBuilder:
    def __init__(self, **kwargs):
        pass

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        return None


model_builder.register_network("task_a2c", lambda **kwargs: _DummyNetBuilder())