"""
Play / evaluate a trained task-learning policy.
1 control step = decimation(4) * dt(1/120s) ≈ 33ms  →  30 steps ≈ 1s

Usage (fixed command):
  python play.py --checkpoint ... --imitation_ckpt ... --expert_ckpt ... --vx 1.0

Scenario A – explicit turn (forward → rotate 180° → forward):
  --phases '[{"vx":1.0,"vy":0,"yaw_deg":0,"steps":60},
             {"vx":0.0,"vy":0,"yaw_deg":180,"steps":60},
             {"vx":1.0,"vy":0,"yaw_deg":0,"steps":60}]'

Scenario B – implicit flip (forward → reverse vx, no yaw cmd):
  --phases '[{"vx":1.0,"vy":0,"yaw_deg":0,"steps":90},
             {"vx":-1.0,"vy":0,"yaw_deg":0,"steps":90}]'

Note: max_steps=0 runs indefinitely (Ctrl+C to stop).
      For phase sequences the loop exits after all phases complete
      regardless of max_steps.
"""
from __future__ import annotations
from pathlib import Path

import argparse
import ast
import json
import yaml

import torch

from isaaclab.app import AppLauncher

PARENT_DIR = Path(__file__).resolve().parents[2]



def _apply_phase(env, phase: dict, device: str, phase_idx: int) -> None:
    """Force all envs to adopt the new command immediately."""
    env.fixed_vx = float(phase["vx"])
    env.fixed_vy = float(phase.get("vy", 0.0))
    # yaw_deg is a delta in degrees added to current_yaw inside _resample_commands
    env.fixed_orientation = float(phase.get("yaw_deg", 0.0))

    all_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    env._resample_commands(all_ids)

    print(
        f"\n[PHASE {phase_idx}] vx={env.fixed_vx:+.2f}  vy={env.fixed_vy:+.2f}"
        f"  yaw_delta={env.fixed_orientation:+.1f}°  duration={phase['steps']} steps",
        flush=True,
    )


def _print_phase_histogram(
    hist: torch.Tensor, phase_idx: int, phase: dict, top_k: int = 5
) -> None:
    """
    hist: (num_active, codebook_size) — cumulative index counts for this phase.
    Prints top-k most used indices per codebook slot.
    """
    num_active, codebook_size = hist.shape
    total = hist.sum().item()
    print(
        f"\n[PHASE {phase_idx} HISTOGRAM] "
        f"vx={phase['vx']:+.2f} yaw_deg={phase.get('yaw_deg', 0.0):+.1f}°  "
        f"total_selections={int(total)}"
    )
    for q in range(num_active):
        counts_q = hist[q]
        k = min(top_k, codebook_size)
        topk_vals, topk_idx = counts_q.topk(k)
        used = (counts_q > 0).sum().item()
        entries = "  ".join(
            f"idx={int(i)}({int(v)/max(total/num_active,1)*100:.1f}%)"
            for i, v in zip(topk_idx.tolist(), topk_vals.tolist())
        )
        print(f"  codebook[{q}]: used={used}/{codebook_size}  top{k}= {entries}")


def main():
    parser = argparse.ArgumentParser("Play Task Learning Policy")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--imitation_ckpt", type=str, required=True)
    parser.add_argument("--expert_ckpt", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--task_goal_dim", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0, help="Max steps to run (0=infinite)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--random_range", type=str)

    # Single fixed command
    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--vy", type=float, default=None)
    # yaw is in degrees (delta added to current robot yaw); kept for backward compat
    parser.add_argument("--yaw", type=float, default=None, help="Yaw delta in degrees")

    # Phase sequence (overrides --vx/--vy/--yaw if provided)
    parser.add_argument(
        "--phases",
        type=str,
        default=None,
        help=(
            'JSON list of phases, e.g. \'[{"vx":1.0,"vy":0,"yaw_deg":0,"steps":200},...]\'. '
            "yaw_deg is the delta orientation in degrees added to current robot yaw. "
            "Scenario A (explicit turn): yaw_deg=180 in phase 2. "
            "Scenario B (implicit flip): vx=-1.0 in phase 2, yaw_deg=0."
        ),
    )

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import g1_hybrid_gym.tasks  # noqa: register envs

    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.rl import rlgames_model_registry  # noqa
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg_task import G1HybridGymEnvTaskCfg
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_navigation_task import G1HybridGymEnvTask
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.rl.wrapper_task_ppo import TaskA2CNetwork
    from g1_hybrid_prior.models.task_learning_block import TaskLearningBlock, TaskCritic

    device = str(args.device if torch.cuda.is_available() else "cpu")

    cfg_path = Path(PARENT_DIR / "train_config/TaskLearning.yaml")
    cfg_path_imitation = Path(PARENT_DIR / "train_config/ImitationLearning.yaml")

    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
    else:
        raise FileNotFoundError(f"[TASK PLAY] Config not found: {cfg_path}")

    env_cfg = G1HybridGymEnvTaskCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device

    env = G1HybridGymEnvTask(cfg=env_cfg, render_mode="human")

    # Parse phases or build a single-phase list from --vx/--vy/--yaw
    phases = None
    if args.phases is not None:
        phases = json.loads(args.phases)
        print(f"[INFO] Phase sequence loaded: {len(phases)} phases")
    elif args.vx is not None or args.vy is not None or args.yaw is not None:
        # Fixed single command — yaw is already in degrees, no conversion needed
        phases = [{
            "vx": args.vx if args.vx is not None else 0.5,
            "vy": args.vy if args.vy is not None else 0.0,
            "yaw_deg": args.yaw if args.yaw is not None else 0.0,
            "steps": args.max_steps if args.max_steps > 0 else 999_999_999,
        }]
        print(f"[INFO] Fixed command: vx={phases[0]['vx']}, vy={phases[0]['vy']}, yaw_deg={phases[0]['yaw_deg']}")

    obs_reset = env.reset()
    if isinstance(obs_reset, tuple):
        obs_reset = obs_reset[0]
    obs_policy = obs_reset["policy"]
    full_obs_dim = obs_policy.shape[-1]
    # Prefer env-reported dims (handles reaching task with task_goal_dim=8 automatically)
    task_goal_dim = getattr(env, "task_goal_dim", args.task_goal_dim)
    s_dim = full_obs_dim - task_goal_dim
    physical_action_dim = 29

    print(f"[INFO] obs_dim={full_obs_dim}, s_dim={s_dim}, task_goal_dim={task_goal_dim}")

    task_block = TaskLearningBlock(
        s_dim=s_dim,
        goal_dim=env.GOAL_DIM,
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

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state = ckpt.get("model", ckpt)
    a2c_keys = {
        k.replace("a2c_network.", "", 1): v
        for k, v in model_state.items()
        if k.startswith("a2c_network.")
    }
    missing, unexpected = a2c_network.load_state_dict(a2c_keys, strict=False)
    print(f"[INFO] Loaded a2c_network: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"Missing (first 10): {missing[:10]}")

    a2c_network.eval()
    task_block.eval()

    codebook_size = task_block.codebook_size
    num_active = task_block.num_active_codebooks
    print(f"[INFO] codebook_size={codebook_size}, num_active_codebooks={num_active}")

    # Phase state
    phase_idx = 0
    phase_step = 0
    phase_hist = torch.zeros(num_active, codebook_size, device=device, dtype=torch.long)
    if phases is not None:
        _apply_phase(env, phases[0], device, 0)

    random_range = ast.literal_eval(args.random_range) if args.random_range else [0, codebook_size]
    low, high = random_range

    obs = obs_reset
    step_count = 0
    episode_rewards = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    completed_episodes = 0
    total_reward_sum = 0.0
    total_length_sum = 0

    print(
        f"[INFO] Starting rollout "
        f"({'deterministic' if args.deterministic else 'random' if args.random else 'stochastic'}) ...",
        flush=True,
    )

    while True:
        if args.max_steps > 0 and step_count >= args.max_steps:
            break

        if isinstance(obs, tuple):
            obs = obs[0]
        obs_flat = obs["policy"] if isinstance(obs, dict) else obs

        s = obs_flat[..., :s_dim]
        g = obs_flat[..., s_dim:]

        with torch.no_grad():
            s_norm = task_block._normalize_s(s)
            hl_out = task_block.high_level(s_norm, g)
            logits = hl_out["logits"]  # (B, num_active, codebook_size)

            if args.deterministic:
                indices = logits.argmax(dim=-1)
            elif args.random:
                indices = torch.randint(low=low, high=high, size=(s.shape[0], num_active), device=device)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                indices = dist.sample()

            physical_action = a2c_network.indices_to_physical_action(s, indices)

        obs, reward, terminated, truncated, _ = env.step(physical_action)
        step_count += 1
        phase_step += 1

        # Accumulate codebook index histogram for current phase
        if phases is not None:
            for q in range(num_active):
                phase_hist[q].scatter_add_(
                    0,
                    indices[:, q],
                    torch.ones(indices.shape[0], device=device, dtype=torch.long),
                )

        # Phase transition
        if phases is not None and phase_step >= phases[phase_idx]["steps"]:
            _print_phase_histogram(phase_hist, phase_idx, phases[phase_idx])
            phase_hist.zero_()
            phase_idx += 1
            phase_step = 0

            if phase_idx < len(phases):
                _apply_phase(env, phases[phase_idx], device, phase_idx)
            else:
                print("\n[INFO] All phases completed.", flush=True)
                if args.max_steps <= 0:
                    break

        episode_rewards += reward
        episode_lengths += 1

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
                        f"  ep={completed_episodes} rew={ep_rew:.2f} len={ep_len}"
                        f"  avg_rew={avg_rew:.2f} avg_len={avg_len:.1f}"
                    )
            episode_rewards[done_ids] = 0.0
            episode_lengths[done_ids] = 0

    # Final histogram for last phase
    if phases is not None and phase_hist.sum() > 0:
        _print_phase_histogram(phase_hist, phase_idx, phases[min(phase_idx, len(phases) - 1)])

    if completed_episodes > 0:
        print(f"\n[SUMMARY] {completed_episodes} episodes  avg_rew={total_reward_sum/completed_episodes:.3f}  avg_len={total_length_sum/completed_episodes:.1f}")
    else:
        print(f"\n[INFO] {step_count} steps, no completed episodes.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
