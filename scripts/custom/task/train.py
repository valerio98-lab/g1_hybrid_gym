# train_task.py
"""
Task Learning training via rl_games PPO (ModelA2CMultiDiscrete).

Usage:
  python train_task.py \
    --imitation_ckpt /path/to/imitation/ckpt_best.pt \
    --num_envs 8192
"""
from __future__ import annotations

import argparse
import datetime
import os

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser("Task Learning — rl_games PPO MultiDiscrete")

    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=10_000)
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
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None)

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # -- Imports after AppLauncher --
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

    env_cfg = G1HybridGymEnvTaskCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device

    base_env = G1HybridGymEnvTask(cfg=env_cfg, render_mode=None)

    obs_reset = base_env.reset()
    if isinstance(obs_reset, tuple):
        obs_reset = obs_reset[0]
    obs_policy = obs_reset["policy"]
    full_obs_dim = obs_policy.shape[-1]
    s_dim = full_obs_dim - args.task_goal_dim
    physical_action_dim = 29

    print(f"[INFO] obs_dim={full_obs_dim}, s_dim={s_dim}, task_goal_dim={args.task_goal_dim}")


    from g1_hybrid_prior.models.task_learning_block import TaskLearningBlock, TaskCritic

    task_block = TaskLearningBlock(
        s_dim=s_dim,
        goal_dim=s_dim,
        task_goal_dim=args.task_goal_dim,
        action_dim=physical_action_dim,
        imitation_ckpt_path=args.imitation_ckpt,
        expert_ckpt_path=args.expert_ckpt, 
    ).to(device)

    critic = TaskCritic(s_dim=s_dim, goal_dim=args.task_goal_dim).to(device)

    a2c_network = TaskA2CNetwork(
        task_block=task_block,
        critic=critic,
        s_dim=s_dim,
        task_goal_dim=args.task_goal_dim,
    )

    codebook_size = task_block.codebook_size
    num_active = task_block.num_active_codebooks
    print(f"[INFO] codebook_size={codebook_size}, num_active_codebooks={num_active}")

    # ---- 3. Wrap env ----
    env = TaskEnvWrapper(
        env=base_env,
        a2c_network=a2c_network,
        s_dim=s_dim,
        num_active_codebooks=num_active,
        codebook_size=codebook_size,
    )

    # ---- 4. Register with rl_games ----
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

    # Register model — but we want rl_games to use the SAME a2c_network
    # we already built (shared with env wrapper).
    # We override build() to return our pre-built network.
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

    # ---- 5. rl_games config ----
    batch_size = args.num_envs * args.horizon
    minibatch_size = args.minibatch_size or batch_size

    rl_config = {
        "params": {
            "seed": 42,
            "algo": {
                "name": "a2c_discrete",
            },
            "model": {
                "name": "wrapper_expert_task_ppo",
            },
            "network": {
                "name": "task_a2c",
            },
            "config": {
                "name": args.run_name,
                "env_name": "isaaclab_task",
                "multi_gpu": False,
                "mixed_precision": False,
                "normalize_input": False,
                "normalize_value": True,
                "value_bootstrap": True,
                "num_actors": args.num_envs,
                "reward_shaper": {"scale_value": 1.0},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": args.lr,
                "lr_schedule": "constant",
                "score_to_win": 1e6,
                "max_epochs": args.max_iterations,
                "save_best_after": 50,
                "save_frequency": 100,
                "grad_norm": 1.0,
                "entropy_coef": args.entropy_coef,
                "truncate_grads": True,
                "e_clip": 0.1,
                "horizon_length": args.horizon,
                "minibatch_size": minibatch_size,
                "mini_epochs": 4,
                "critic_coef": 0.5,
                "clip_value": True,
                "seq_length": 4,
                "bounds_loss_coef": 10.0,
            },
        },
    }

    # ---- 6. Train ----
    os.makedirs(args.log_dir, exist_ok=True)

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