"""
Task Learning training via rl_games PPO (ModelA2CMultiDiscrete).
"""

from __future__ import annotations
from pathlib import Path

import argparse
import datetime
import os
import yaml

from isaaclab.app import AppLauncher

PARENT_DIR = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser("Task Learning — rl_games PPO MultiDiscrete")

    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10_000)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--log_dir", type=str,
        default=f"./logs/task_learning/{datetime.datetime.now().strftime('%d_%m_%Y_%H%M%S')}",
    )
    parser.add_argument("--run_name", type=str, default="g1_vel_tracking")
    parser.add_argument("--imitation_ckpt", type=str, required=True)
    parser.add_argument("--expert_ckpt", type=str, required=True)
    parser.add_argument("--task_goal_dim", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--minibatch_size", type=int, default=32768)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None)

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    import g1_hybrid_gym.tasks  # noqa: register envs

    from rl_games.torch_runner import Runner
    from rl_games.common import env_configurations, vecenv
    from rl_games.algos_torch import model_builder
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.rl import rlgames_model_registry  # noqa: register models+networks

    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg_task import G1HybridGymEnvTaskCfg
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_navigation_task import G1HybridGymEnvTask
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.rl.wrapper_task_ppo import TaskPolicyWrapper, TaskA2CNetwork
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.task_env_wrapper import TaskEnvWrapper

    device = str(args.device if torch.cuda.is_available() else "cpu")

    log_dir = str(f"{args.log_dir}_{args.experiment_name}")
    env_cfg = G1HybridGymEnvTaskCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device

    base_env = G1HybridGymEnvTask(cfg=env_cfg, render_mode=None)

    batch_size = args.num_envs * args.horizon
    minibatch_size = args.minibatch_size or batch_size

    cfg_path = Path(PARENT_DIR / "train_config/TaskLearning.yaml")
    cfg_path_imitation = Path(PARENT_DIR / "train_config/ImitationLearning.yaml")

    if cfg_path.exists(): 
        cfg = yaml.safe_load(cfg_path.read_text())
    else: 
        raise Exception("[TASK TRAIN]: Config file not found", cfg_path)
    rl_config = cfg["task_learning_policy"]["rl_config"]

    rl_config["params"]["config"]["name"] = args.run_name
    rl_config["params"]["config"]["log_dir"] = log_dir
    rl_config["params"]["config"]["train_dir"] = log_dir
    rl_config["params"]["config"]["num_actors"] = args.num_envs
    rl_config["params"]["config"]["max_epochs"] = args.max_iterations
    rl_config["params"]["config"]["horizon_length"] = args.horizon
    rl_config["params"]["config"]["minibatch_size"] = args.minibatch_size
    rl_config["params"]["config"]["learning_rate"] = args.lr
    rl_config["params"]["config"]["entropy_coef"] = args.entropy_coef
    rl_config["params"]["config"]["score_to_win"] = float(cfg["task_learning_policy"]["rl_config"]["params"]["config"]["score_to_win"])

    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
    env_rl = RlGamesVecEnvWrapper(base_env, rl_device=device, clip_obs=rl_config["params"]["env"]["clip_observations"], clip_actions=rl_config["params"]["env"]["clip_actions"])

    obs_reset = base_env.reset()
    if isinstance(obs_reset, tuple):
        obs_reset = obs_reset[0]
    obs_policy = obs_reset["policy"]
    full_obs_dim = obs_policy.shape[-1]
    # Prefer env-reported dims (handles reaching task with task_goal_dim=8 automatically)
    task_goal_dim = getattr(base_env, "task_goal_dim", args.task_goal_dim)
    s_dim = full_obs_dim - task_goal_dim
    physical_action_dim = 29

    print(f"[INFO] obs_dim={full_obs_dim}, s_dim={s_dim}, task_goal_dim={task_goal_dim}")


    from g1_hybrid_prior.models.task_learning_block import TaskLearningBlock, TaskCritic

    task_block = TaskLearningBlock(
        s_dim=s_dim,
        goal_dim=base_env.GOAL_DIM,
        task_goal_dim=task_goal_dim,
        action_dim=physical_action_dim,
        imitation_ckpt_path=args.imitation_ckpt,
        expert_ckpt_path=args.expert_ckpt, 
        cfg_path=cfg_path,
        imitation_cfg_path=cfg_path_imitation,

    ).to(device)

    critic = TaskCritic(s_dim=s_dim, goal_dim=args.task_goal_dim, cfg_path=cfg_path).to(device)

    a2c_network = TaskA2CNetwork(
        task_block=task_block,
        critic=critic,
        s_dim=s_dim,
        task_goal_dim=args.task_goal_dim,
    )

    codebook_size = task_block.codebook_size
    num_active = task_block.num_active_codebooks
    print(f"[INFO] codebook_size={codebook_size}, num_active_codebooks={num_active}")

    env = TaskEnvWrapper(
        env=env_rl,
        a2c_network=a2c_network,
        s_dim=s_dim,
        num_active_codebooks=num_active,
        codebook_size=codebook_size,
    )

    vecenv.register(
        "ISAACLAB_TASK",
        lambda config_name, num_actors, **kwargs: env,
    )
    env_configurations.register(
        "isaaclab_task",
        {
            "vecenv_type": "ISAACLAB_TASK",
            "env_creator": lambda **kwargs: env,
        },
    )
    class PrebuiltTaskWrapper(TaskPolicyWrapper):
        """Uses the pre-built a2c_network instead of constructing a new one."""
        def build(self, config):
            value_size = config.get("value_size", 1)
            normalize_value = config["normalize_value"]
            normalize_input = config["normalize_input"]
            obs_shape = config["input_shape"]

            return self.Network(
                a2c_network,
                obs_shape=obs_shape,
                normalize_value=normalize_value,
                normalize_input=normalize_input,
                value_size=value_size,
            )

    model_builder.register_model(
        "wrapper_expert_task_ppo",
        lambda network, **kwargs: PrebuiltTaskWrapper(network),
    )


    os.makedirs(args.log_dir , exist_ok=True)

    runner = Runner(IsaacAlgoObserver())
    runner.load(rl_config)
    runner.reset()

    runner.run(
        {
            "train": True,
            "play": False,
            "checkpoint": args.resume,
            "sigma": None,
        },
    )

    print("[INFO] Training complete.")
    base_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()