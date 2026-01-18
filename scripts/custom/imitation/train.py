# g1_hybrid_gym/train_imitation.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml

from isaaclab.app import AppLauncher

import g1_hybrid_gym.tasks  # noqa: F401

from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_ppo import G1HybridGymEnvPPO

from g1_hybrid_prior.models.expert_policy import ExpertPolicy
from g1_hybrid_prior.models.hybrid_imitation_block import ImitationBlock, LossWeights
from g1_hybrid_prior.trainers.imitation_trainer import ImitationTrainer, TrainerCfg


def _reset_env(env) -> Dict[str, torch.Tensor]:
    # Gymnasium compat: reset can return (obs, info)
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
        return obs
    return out


def _step_env(env, action: torch.Tensor):
    # Gymnasium compat: step returns 5-tuple
    return env.step(action)


def _split_obs(obs_policy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # obs_policy: (N, s_dim + goal_dim), with s_dim == goal_dim
    full_dim = obs_policy.shape[-1]
    if full_dim % 2 != 0:
        raise RuntimeError(f"Expected even obs dim, got {full_dim}")
    s_dim = full_dim // 2
    s = obs_policy[..., :s_dim]
    goal = obs_policy[..., s_dim:]
    return s, goal


@torch.no_grad()
def _get_mu_expert(expert: ExpertPolicy, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    mu, _, _ = expert(s, goal)
    return mu


def main():
    parser = argparse.ArgumentParser("Train imitation block (online distillation)")

    # --- your args
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--log_dir", type=str, default="runs/imitation")
    parser.add_argument("--run_name", type=str, default="g1_hybrid_imitation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--imitation_cfg", type=str, required=True,
                        help="Path to YAML containing imitation_learning_policy section")
    parser.add_argument("--expert_checkpoint", type=str, default=None,
                        help="(Optional) path to expert checkpoint if you load weights manually")

    AppLauncher.add_app_launcher_args(parser)

    # parse
    args, _unknown = parser.parse_known_args()

    # launch omniverse app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- load imitation yaml
    imitation_cfg_path = Path(args.imitation_cfg)
    with open(imitation_cfg_path, "r") as f:
        net_cfg = yaml.safe_load(f)

    if "imitation_learning_policy" not in net_cfg:
        raise RuntimeError(
            f"Config {imitation_cfg_path} must contain top-level key 'imitation_learning_policy'"
        )

    env_cfg = G1HybridGymEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)

    # create env
    env = G1HybridGymEnvPPO(cfg=env_cfg, render_mode=None)

    # infer dims from env
    obs = _reset_env(env)
    obs_policy = obs["policy"]
    obs_dim = obs_policy.shape[-1]
    if obs_dim % 2 != 0:
        raise RuntimeError(f"Bad obs_dim={obs_dim}. Expected [s_cur, goal] with same dim.")

    s_dim = obs_dim // 2
    goal_dim = obs_dim // 2
    action_dim = int(env_cfg.action_space)  

    # expert
    expert = ExpertPolicy(obs_dim=s_dim, goal_dim=goal_dim, action_dim=action_dim, device=str(device))
    expert.eval()

    if args.expert_checkpoint is not None:
        ckpt = torch.load(args.expert_checkpoint, map_location=device)
        if isinstance(ckpt, dict):
            state_dict = (
                ckpt.get("model")
                or ckpt.get("state_dict")
                or ckpt.get("policy")
                or ckpt
            )
        else:
            state_dict = ckpt
        expert.load_state_dict(state_dict, strict=True)
 
    use_expert_decoder = bool(net_cfg["imitation_learning_policy"].get("use_expert_decoder", True))
    expert_decoder = expert.decoder if use_expert_decoder else None

    model = ImitationBlock(
        s_dim=s_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        net_cfg=net_cfg,
        expert_decoder=expert_decoder,
    ).to(device)

    # loss weights (you can tune later)
    loss_w = LossWeights(action=1.0, mm=0.1, reg=0.0, vq=1.0)

    trainer_cfg = TrainerCfg(
        device=str(device),
        use_amp=args.use_amp,
        ckpt_dir=f"{args.log_dir}/ckpts",
    )

    trainer = ImitationTrainer(
        model=model,
        loss_weights=loss_w,
        cfg=trainer_cfg,
        log_dir=args.log_dir,
        run_name=args.run_name,
    )

    obs = _reset_env(env)
    done_prev = torch.ones((env.unwrapped.num_envs,), device=device, dtype=torch.bool)
    prev_ref_idx = None
    wrap_prev = torch.zeros_like(done_prev)

    # grab current ref idx (after reset) as initial previous (no reg anyway because done_prev=True)
    if hasattr(env.unwrapped, "ref_frame_idx"):
        prev_ref_idx = env.unwrapped.ref_frame_idx.clone()
    else:
        raise RuntimeError("Env must expose env.unwrapped.ref_frame_idx for definitive temporal masking.")
        
    for it in range(args.steps):
        obs_policy = obs["policy"]
        s, goal = _split_obs(obs_policy)
        valid_prev = (~done_prev) & (~wrap_prev)

        mu_expert = _get_mu_expert(expert, s, goal)

        stats = trainer.train_step({"s": s, "goal": goal, "a_expert": mu_expert, "valid_prev": valid_prev})

        # teacher forcing rollout
        obs, rew, terminated, truncated, extras = _step_env(env, mu_expert)
        
        done_prev = (terminated | truncated).to(device=device, dtype=torch.bool)
        new_ref_idx = env.unwrapped.ref_frame_idx
        wrap_prev = torch.zeros_like(done_prev)
        if prev_ref_idx is None:
            prev_ref_idx = new_ref_idx.clone()
        else:
            # wrap if it jumped to 0 without being a reset transition
            wrap_prev = (new_ref_idx == 0) & (prev_ref_idx != 0) & (~done_prev)
            prev_ref_idx = new_ref_idx.clone()

        if (it + 1) % 1000 == 0:
            print(
                f"[it={it+1}] loss_total={stats['loss_total']:.4f} "
                f"loss_action={stats['loss_action']:.4f} loss_mm={stats['loss_mm']:.4f} "
                f"loss_commit={stats.get('loss_commit', 0.0):.4f}"
            )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
