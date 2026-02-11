"""
Play / evaluate a trained task-learning policy.

Usage:
  python play_task.py \
    --checkpoint /path/to/last_g1_vel_tracking.pth \
    --imitation_ckpt /path/to/imitation/ckpt_best.pt \
    --expert_ckpt /path/to/expert.pth \
    --num_envs 16 \
    --headless  (optional)
"""
from __future__ import annotations

import argparse
import torch

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser("Play Task Learning Policy")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to rl_games .pth checkpoint")
    parser.add_argument("--imitation_ckpt", type=str, required=True)
    parser.add_argument("--expert_ckpt", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--task_goal_dim", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=5000, help="Max steps to run (0=infinite)")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling")

    # Fixed velocity command (optional, otherwise random resampling)
    parser.add_argument("--vx", type=float, default=None, help="Override vx command")
    parser.add_argument("--vy", type=float, default=None, help="Override vy command")
    parser.add_argument("--omega", type=float, default=None, help="Override omega command")

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import g1_hybrid_gym.tasks  # noqa: register envs

    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg_task import G1HybridGymEnvTaskCfg
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_navigation_task import G1HybridGymEnvTask
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.rl.wrapper_task_ppo import TaskA2CNetwork

    from g1_hybrid_prior.models.task_learning_block import TaskLearningBlock, TaskCritic

    device = str(args.device if torch.cuda.is_available() else "cpu")

    env_cfg = G1HybridGymEnvTaskCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device

    env = G1HybridGymEnvTask(cfg=env_cfg, render_mode="human")

    obs_reset = env.reset()
    if isinstance(obs_reset, tuple):
        obs_reset = obs_reset[0]
    obs_policy = obs_reset["policy"]
    full_obs_dim = obs_policy.shape[-1]
    s_dim = full_obs_dim - args.task_goal_dim
    physical_action_dim = 29

    print(f"[INFO] obs_dim={full_obs_dim}, s_dim={s_dim}, task_goal_dim={args.task_goal_dim}")

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

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state = ckpt.get("model", ckpt)


    a2c_keys = {k.replace("a2c_network.", "", 1): v 
                for k, v in model_state.items() 
                if k.startswith("a2c_network.")}

    missing, unexpected = a2c_network.load_state_dict(a2c_keys, strict=False)
    print(f"[INFO] Loaded a2c_network weights. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected (first 10): {unexpected[:10]}")

    a2c_network.eval()
    task_block.eval()

    codebook_size = task_block.codebook_size
    num_active = task_block.num_active_codebooks
    print(f"[INFO] codebook_size={codebook_size}, num_active_codebooks={num_active}")

    fixed_cmd = None
    if args.vx is not None or args.vy is not None or args.omega is not None:
        vx = args.vx if args.vx is not None else 0.5
        vy = args.vy if args.vy is not None else 0.0
        omega = args.omega if args.omega is not None else 0.0
        fixed_cmd = torch.tensor([vx, vy, omega], device=device).unsqueeze(0).expand(args.num_envs, -1)
        print(f"[INFO] Fixed velocity command: vx={vx}, vy={vy}, omega={omega}")

    obs = obs_reset
    step_count = 0
    episode_rewards = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    completed_episodes = 0
    total_reward_sum = 0.0
    total_length_sum = 0

    print(f"[INFO] Starting rollout ({'deterministic' if args.deterministic else 'stochastic'})...")

    while True:
        if args.max_steps > 0 and step_count >= args.max_steps:
            break

        # Override velocity commands if fixed
        if fixed_cmd is not None:
            env.vel_cmd[:] = fixed_cmd

        # Get observation
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict):
            obs_flat = obs["policy"]
        else:
            obs_flat = obs

        s = obs_flat[..., :s_dim]
        g = obs_flat[..., s_dim:]

        # Forward pass through high-level policy
        with torch.no_grad():
            s_norm = task_block._normalize_s(s)
            hl_out = task_block.high_level(s_norm, g)
            logits = hl_out["logits"]  # (B, num_active, codebook_size)

            if args.deterministic:
                indices = logits.argmax(dim=-1)  # (B, num_active)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                indices = dist.sample()

            # Decode to physical action
            physical_action = a2c_network.indices_to_physical_action(s, indices)

        # Step environment
        obs, reward, terminated, truncated, extras = env.step(physical_action)

        episode_rewards += reward
        episode_lengths += 1
        step_count += 1

        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_ids:
                completed_episodes += 1
                ep_rew = episode_rewards[idx].item()
                ep_len = episode_lengths[idx].item()
                total_reward_sum += ep_rew
                total_length_sum += ep_len

                if completed_episodes % 10 == 0 or completed_episodes <= 5:
                    avg_rew = total_reward_sum / completed_episodes
                    avg_len = total_length_sum / completed_episodes
                    print(
                        f"  Episode {completed_episodes}: "
                        f"reward={ep_rew:.2f}, length={ep_len}, "
                        f"avg_reward={avg_rew:.2f}, avg_length={avg_len:.1f}"
                    )

            episode_rewards[done_ids] = 0.0
            episode_lengths[done_ids] = 0

    if completed_episodes > 0:
        print(f"\n[SUMMARY] {completed_episodes} episodes completed in {step_count} steps")
        print(f"  Average reward:  {total_reward_sum / completed_episodes:.3f}")
        print(f"  Average length:  {total_length_sum / completed_episodes:.1f}")
    else:
        print(f"\n[INFO] {step_count} steps completed, no episodes finished.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()