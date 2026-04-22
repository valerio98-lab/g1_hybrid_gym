"""
Play / evaluate the task-learning policy on the arm reaching task.
1 control step = decimation(4) * dt(1/120s) ≈ 33ms  →  30 steps ≈ 1s

Usage (free resampling):
  python play_reaching.py --checkpoint ... --imitation_ckpt ... --expert_ckpt ...

Scenario 1 – stand still + reach:
  --phases '[{"vx":0.0,"vy":0,"steps":180}]'

Scenario 2 – walk + reach:
  --phases '[{"vx":0.8,"vy":0,"steps":180}]'

Scenario 3 – walk with arms as wings (fixed targets):
  --phases '[{"vx":0.8,"vy":0,
              "ee_targets":[[-0.05,-0.65,1.0],[-0.05,0.65,1.0]],
              "steps":180}]'

Multi-phase example (stand → walk → wings):
  --phases '[{"vx":0.0,"vy":0,"steps":90},
             {"vx":0.8,"vy":0,"steps":90},
             {"vx":0.8,"vy":0,"ee_targets":[[-0.05,-0.65,1.0],[-0.05,0.65,1.0]],"steps":90}]'

Note: max_steps=0 runs indefinitely (Ctrl+C to stop).
      ee_targets: list of [x, y, z] per arm EE in robot body frame.
      Order matches reaching_arm_ee_names in config (default: right hand, left hand).
      Set to null in a phase to resume free resampling: {"vx":0.8,"ee_targets":null,"steps":90}
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


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _apply_phase(env, phase: dict, device: str, phase_idx: int) -> None:
    """Force all envs to adopt the new command immediately."""
    env.fixed_vx = float(phase["vx"])
    env.fixed_vy = float(phase.get("vy", 0.0))

    ee_targets = phase.get("ee_targets", None)
    if ee_targets is not None:
        env.fixed_ee_targets = [tuple(t) for t in ee_targets]
    else:
        env.fixed_ee_targets = None

    all_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    env._resample_commands(all_ids)

    ee_str = str(ee_targets) if ee_targets is not None else "free resample"
    print(
        f"\n[PHASE {phase_idx}] vx={env.fixed_vx:+.2f}  vy={env.fixed_vy:+.2f}"
        f"  ee_targets={ee_str}  duration={phase['steps']} steps",
        flush=True,
    )


def _print_phase_histogram(
    hist: torch.Tensor, phase_idx: int, phase: dict, top_k: int = 5
) -> None:
    """hist: (num_active, codebook_size) — cumulative index counts for this phase."""
    num_active, codebook_size = hist.shape
    total = hist.sum().item()
    ee_str = "fixed" if phase.get("ee_targets") else "free"
    print(
        f"\n[PHASE {phase_idx} HISTOGRAM] "
        f"vx={phase['vx']:+.2f} ee={ee_str}  total_selections={int(total)}"
    )
    for q in range(num_active):
        counts_q = hist[q]
        k = min(top_k, codebook_size)
        topk_vals, topk_idx = counts_q.topk(k)
        used = (counts_q > 0).sum().item()
        entries = "  ".join(
            f"idx={int(i)}({int(v) / max(total / num_active, 1) * 100:.1f}%)"
            for i, v in zip(topk_idx.tolist(), topk_vals.tolist())
        )
        print(f"  codebook[{q}]: used={used}/{codebook_size}  top{k}= {entries}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Play Task Learning Policy – Reaching Task")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--imitation_ckpt", type=str, required=True)
    parser.add_argument("--expert_ckpt", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=0, help="Max steps (0=infinite)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--random_range", type=str)

    # Simple fixed overrides (no phases)
    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--vy", type=float, default=None)

    # Phase sequence
    parser.add_argument(
        "--phases",
        type=str,
        default=None,
        help=(
            'JSON list of phases. Each phase: {"vx":float,"vy":float,"steps":int,'
            '"ee_targets":[[x,y,z],[x,y,z]] or null}. '
            "ee_targets null = free resampling. "
            "Wings example: ee_targets=[[-0.05,-0.65,1.0],[-0.05,0.65,1.0]]"
        ),
    )

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import g1_hybrid_gym.tasks  # noqa: register envs

    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.rl import rlgames_model_registry  # noqa
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg_reaching_task import G1HybridGymEnvReachingCfg
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_reaching_task import G1HybridGymEnvReaching
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.rl.wrapper_task_ppo import TaskA2CNetwork
    from g1_hybrid_prior.models.task_learning_block import TaskLearningBlock, TaskCritic

    device = str(args.device if torch.cuda.is_available() else "cpu")

    cfg_path = Path(PARENT_DIR / "train_config/TaskLearning.yaml")
    cfg_path_imitation = Path(PARENT_DIR / "train_config/ImitationLearning.yaml")

    if not cfg_path.exists():
        raise FileNotFoundError(f"[REACHING PLAY] Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    env_cfg = G1HybridGymEnvReachingCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device

    env = G1HybridGymEnvReaching(cfg=env_cfg, render_mode="human")

    # task_goal_dim comes from the env (2 + num_arm_ee * 3 = 8)
    task_goal_dim = env.task_goal_dim
    s_dim = env.s_dim
    physical_action_dim = 29

    print(f"[INFO] task_goal_dim={task_goal_dim}, s_dim={s_dim}, arm_ee={env.arm_ee_names}")

    # Parse phases or build one from --vx/--vy
    phases = None
    if args.phases is not None:
        phases = json.loads(args.phases)
        print(f"[INFO] Phase sequence loaded: {len(phases)} phases")
    elif args.vx is not None or args.vy is not None:
        phases = [{
            "vx": args.vx if args.vx is not None else 0.0,
            "vy": args.vy if args.vy is not None else 0.0,
            "ee_targets": None,
            "steps": args.max_steps if args.max_steps > 0 else 999_999_999,
        }]
        print(f"[INFO] Fixed command: vx={phases[0]['vx']}, vy={phases[0]['vy']}")

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

    critic = TaskCritic(s_dim=s_dim, goal_dim=task_goal_dim, cfg_path=cfg_path).to(device)

    a2c_network = TaskA2CNetwork(
        task_block=task_block,
        critic=critic,
        s_dim=s_dim,
        task_goal_dim=task_goal_dim,
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
        print(f"  Missing (first 10): {missing[:10]}")

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

    obs_reset = env.reset()
    if isinstance(obs_reset, tuple):
        obs_reset = obs_reset[0]
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
            logits = hl_out["logits"]

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

        # Accumulate codebook histogram
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

    # Final histogram for last incomplete phase
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
