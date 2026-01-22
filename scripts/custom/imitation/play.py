from __future__ import annotations

import argparse
import torch
import yaml
from pathlib import Path
from typing import Tuple
from collections import OrderedDict

from isaaclab.app import AppLauncher

from g1_hybrid_prior.models.expert_policy import ExpertPolicy
from g1_hybrid_prior.models.hybrid_imitation_block import ImitationBlock


def _split_obs(obs_policy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    full_dim = obs_policy.shape[-1]
    if full_dim % 2 != 0:
        raise RuntimeError(f"Expected even obs dim, got {full_dim}")
    s_dim = full_dim // 2
    s = obs_policy[..., :s_dim]
    goal = obs_policy[..., s_dim:]
    return s, goal


def _load_expert(expert: ExpertPolicy, ckpt_path: str, device: torch.device) -> None:
    print(f"[INFO] Loading expert from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k: v for k, v in sd.items() if k.startswith("a2c_network.")}
    sd = OrderedDict((k.replace("a2c_network.", "", 1), v) for k, v in sd.items())

    missing, unexpected = expert.load_state_dict(sd, strict=False)
    print(f"[INFO] Expert loaded. missing={len(missing)} unexpected={len(unexpected)}")


def main():
    parser = argparse.ArgumentParser("Play trained imitation policy")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the student (imitation) checkpoint (ckpt_*.pt or raw state_dict).",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments to simulate (default 1 for viz).",
    )

    # Expert checkpoint for (A) expert decoder (if enabled) AND (B) debug comparison prints
    parser.add_argument(
        "--expert_checkpoint",
        type=str,
        default="/home/valerio/g1_hybrid_gym/logs/rl_games/g1_hybrid_expert_PPO/2026-01-15_17-39-38/nn/g1_hybrid_expert_PPO.pth",
        help="Path to the expert PPO checkpoint (rl_games .pth).",
    )
    parser.add_argument("--control", choices=["student", "expert"], default="student",
                        help="Whether to control the env with the student or expert policy.")

    # Isaac Lab standard args
    AppLauncher.add_app_launcher_args(parser)
    args, _unknown = parser.parse_known_args()

    if args.headless is None:
        args.headless = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    print("[INFO] Simulation App started. Importing modules...")

    import g1_hybrid_gym.tasks  # noqa: F401
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_imitation import (
        G1HybridGymEnvImitation,
    )

    device = torch.device(args.device)

    # Environment config
    env_cfg = G1HybridGymEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)

    print(f"[INFO] Creating environment with {args.num_envs} envs...")
    env = G1HybridGymEnvImitation(
        cfg=env_cfg, render_mode="rgb_array" if args.headless else None
    )

    # First reset to infer dims
    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):
        obs_dict = obs_dict[0]

    obs_policy = obs_dict["policy"]
    obs_dim = int(obs_policy.shape[-1])
    s_dim = obs_dim // 2
    goal_dim = obs_dim // 2

    # Infer action dim
    try:
        if hasattr(env, "single_action_space"):
            action_dim = int(env.single_action_space.shape[0])
        else:
            action_dim = int(env.action_space.shape[0])
    except Exception:
        action_dim = int(env_cfg.action_space)

    print(f"[INFO] Dimensions: S={s_dim}, Goal={goal_dim}, Action={action_dim}")

    # Read YAML to decide if we must use expert decoder inside student model
    cfg_path = Path("/home/valerio/g1_hybrid_prior/config/ImitationLearning.yaml")
    net_cfg = yaml.safe_load(cfg_path.read_text())
    use_expert_decoder = bool(
        net_cfg["imitation_learning_policy"].get("use_expert_decoder", False)
    )

    print(f"[INFO] use_expert_decoder from yaml = {use_expert_decoder}")
    
    EXPERT_CKPT = "/home/valerio/g1_hybrid_gym/logs/rl_games/g1_hybrid_expert_PPO/2026-01-15_17-39-38/nn/g1_hybrid_expert_PPO.pth"

    expert = ExpertPolicy(obs_dim=s_dim, goal_dim=goal_dim, action_dim=action_dim, device=str(device)).to(device)
    expert.eval()
    expert.load_from_rlgames(EXPERT_CKPT, strict=False, load_rms=True, enable_rms=False, clip=5.0)

    expert_decoder = expert.decoder if use_expert_decoder else None

    # --- Build student ---
    model = ImitationBlock(
        s_dim=s_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        expert_decoder=expert_decoder,
    ).to(device)
    model.eval()

    # --- Load student checkpoint ---
    print(f"[INFO] Loading student checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # strict=True = fail fast if mismatch
    model.load_state_dict(state_dict, strict=True)
    print("[INFO] Student loaded with strict=True (OK).")


    print("[INFO] Starting play loop...")

    with torch.no_grad():
        obs_dict = env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]

        t = 0
        while simulation_app.is_running():
            t += 1
            obs_policy = obs_dict["policy"].to(device=device, dtype=torch.float32)
            s, goal = _split_obs(obs_policy)

            # normalize exactly like training
            full = torch.cat([s, goal], dim=-1)
            full_n = expert.obs_rms.normalize(full, clip=5.0)
            s_n = full_n[..., :s_dim]
            goal_n = full_n[..., s_dim:]

            # --- compute both ---
            mu_expert, _, _ = expert(s_n, goal_n)

            out = model(s_n, goal_n)
            a_student = out["a_hat"] if isinstance(out, dict) else out

            # --- choose control source ---
            if args.control == "expert":
                actions = mu_expert
            else:
                actions = a_student

            actions = actions.to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)

            # --- debug print every 50 steps (always compares student vs expert) ---
            if (t % 50) == 0:
                mse = ((a_student - mu_expert) ** 2).mean().item()
                cos = torch.nn.functional.cosine_similarity(a_student, mu_expert, dim=-1).mean().item()
                print(
                    f"[dbg] t={t} mse(student,expert)={mse:.6f} cos={cos:.4f}  "
                    f"rms_student={a_student.pow(2).mean().sqrt().item():.3f} "
                    f"rms_exp={mu_expert.pow(2).mean().sqrt().item():.3f}"
                )

            if actions.dim() == 1:
                actions = actions.unsqueeze(0)

            obs_dict, rew, terminated, truncated, extras = env.step(actions)


    print("[INFO] Closing...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
