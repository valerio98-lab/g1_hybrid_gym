from __future__ import annotations

import argparse
import torch
import yaml
from pathlib import Path
from typing import Tuple

from isaaclab.app import AppLauncher

OBS_CLIP = 5.0
PARENT_DIR = Path(__file__).parents[2]


def _split_obs(obs_policy: torch.Tensor, CUR_OBS_DIM: int) -> Tuple[torch.Tensor, torch.Tensor]:
    s = obs_policy[..., :CUR_OBS_DIM]
    goal = obs_policy[..., CUR_OBS_DIM:]
    return s, goal


def main():
    parser = argparse.ArgumentParser("Play trained imitation policy")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the student (imitation) checkpoint.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--expert_checkpoint", type=str, required=True,
                        help="Path to the expert PPO checkpoint (rl_games .pth).")
    parser.add_argument("--control", choices=["student", "expert"], default="student")

    AppLauncher.add_app_launcher_args(parser)
    args, _unknown = parser.parse_known_args()

    if args.headless is None:
        args.headless = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    print("[INFO] Simulation App started. Importing modules...")

    import g1_hybrid_gym.tasks  # noqa: F401
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_imitation import G1HybridGymEnvImitation
    from g1_hybrid_prior.models.expert_policy import ExpertPolicy
    from g1_hybrid_prior.models.hybrid_imitation_block import ImitationBlock

    device = torch.device(args.device)

    env_cfg = G1HybridGymEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)

    print(f"[INFO] Creating environment with {args.num_envs} envs...")
    env = G1HybridGymEnvImitation(cfg=env_cfg, render_mode="rgb_array" if args.headless else None)

    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):
        obs_dict = obs_dict[0]

    # Use env-exposed dims (consistent with train script)
    CUR_OBS_DIM = env.CUR_OBS_DIM
    GOAL_DIM = env.GOAL_DIM

    try:
        action_dim = int(env.single_action_space.shape[0]) if hasattr(env, "single_action_space") else int(env.action_space.shape[0])
    except Exception:
        action_dim = int(env_cfg.action_space)

    print(f"[INFO] Dimensions: S={CUR_OBS_DIM}, Goal={GOAL_DIM}, Action={action_dim}")

    cfg_path = Path(PARENT_DIR / "train_config/ImitationLearning.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    net_cfg = yaml.safe_load(cfg_path.read_text())
    use_expert_decoder = bool(net_cfg["imitation_learning_policy"].get("use_expert_decoder", False))
    print(f"[INFO] use_expert_decoder={use_expert_decoder}")

    expert = ExpertPolicy(obs_dim=CUR_OBS_DIM, goal_dim=GOAL_DIM, action_dim=action_dim, device=str(device)).to(device)
    expert.eval()
    expert.load_from_rlgames(args.expert_checkpoint, strict=False, load_rms=True, enable_rms=False, clip=OBS_CLIP)

    expert_decoder = expert.decoder if use_expert_decoder else None

    model = ImitationBlock(
        s_dim=CUR_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=action_dim,
        expert_decoder=expert_decoder,
        net_cfg_path=cfg_path,
    ).to(device)
    model.eval()

    print(f"[INFO] Loading student checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt
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
            s, goal = _split_obs(obs_policy, CUR_OBS_DIM)

            full = torch.cat([s, goal], dim=-1)
            full_n = expert.obs_rms.normalize(full, clip=OBS_CLIP)
            s_n = full_n[..., :CUR_OBS_DIM]
            goal_n = full_n[..., CUR_OBS_DIM:]

            mu_expert, _, _ = expert(s_n, goal_n)
            out = model(s_n, goal_n)
            a_student = out["a_hat"] if isinstance(out, dict) else out

            actions = (mu_expert if args.control == "expert" else a_student)
            actions = actions.to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)

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