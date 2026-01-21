from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import yaml
from tqdm import tqdm
from collections import OrderedDict



from isaaclab.app import AppLauncher


def _reset_env(env) -> Dict[str, torch.Tensor]:
    # Gymnasium compat
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
        return obs
    return out

def _step_env(env, action: torch.Tensor):
    return env.step(action)

def _split_obs(obs_policy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    full_dim = obs_policy.shape[-1]
    if full_dim % 2 != 0:
        raise RuntimeError(f"Expected even obs dim, got {full_dim}")
    s_dim = full_dim // 2
    s = obs_policy[..., :s_dim]
    goal = obs_policy[..., s_dim:]
    return s, goal

@torch.no_grad()
def _get_mu_expert(expert: Any, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    mu, _, _ = expert(s, goal)
    return mu


def main():
    parser = argparse.ArgumentParser("Train imitation block (online distillation)")

    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--log_dir", type=str, default="./logs/imitation")
    parser.add_argument("--run_name", type=str, default="g1_hybrid_imitation")
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--expert_checkpoint", type=str, default=None,
                        help="(Optional) path to expert checkpoint if you load weights manually")
    parser.add_argument("--debug_temporal", action="store_true",
                        help="Enable extra temporal consistency checks")

    # Aggiunge argomenti per Isaac Sim
    AppLauncher.add_app_launcher_args(parser)
    args, _unknown = parser.parse_known_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    print("[INFO] Simulation App started. Importing tasks and models...")
    
    import g1_hybrid_gym.tasks  # noqa: F401
    
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_imitation import G1HybridGymEnvImitation
    
    from g1_hybrid_prior.models.expert_policy import ExpertPolicy
    from g1_hybrid_prior.models.hybrid_imitation_block import ImitationBlock
    from g1_hybrid_prior.trainers.imitation_trainer import ImitationTrainer, TrainerCfg, LossWeights


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    env_cfg = G1HybridGymEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)

    print(f"[INFO] Creating environment on {device}...")
    env = G1HybridGymEnvImitation(cfg=env_cfg, render_mode=None)

    obs = _reset_env(env)
    obs_policy = obs["policy"]
    obs_dim = obs_policy.shape[-1]
    
    if obs_dim % 2 != 0:
        raise RuntimeError(f"Bad obs_dim={obs_dim}. Expected [s_cur, goal] with same dim.")

    s_dim = obs_dim // 2
    goal_dim = obs_dim // 2
    
    try:
        action_dim = int(env.action_space.shape[0])
        # Controllo se è uno spazio vettorizzato (num_envs, action_dim)
        if hasattr(env, "single_action_space"):
             action_dim = int(env.single_action_space.shape[0])
    except:
         action_dim = int(env_cfg.action_space)

    print(f"[INFO] Dimensions: S={s_dim}, Goal={goal_dim}, Action={action_dim}")

    expert = ExpertPolicy(obs_dim=s_dim, goal_dim=goal_dim, action_dim=action_dim, device=str(device))
    expert.eval()

    if args.expert_checkpoint is not None:
        print(f"[INFO] Loading expert from {args.expert_checkpoint}")
        ckpt = torch.load(args.expert_checkpoint, map_location=device, weights_only=False)

        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        sd = {k: v for k, v in sd.items() if k.startswith("a2c_network.")}

        sd = OrderedDict((k.replace("a2c_network.", "", 1), v) for k, v in sd.items())

        missing, unexpected = expert.load_state_dict(sd, strict=False)
        print(f"[INFO] Expert loaded. missing={len(missing)} unexpected={len(unexpected)}")

    
    cfg_path = Path("/home/valerio/g1_hybrid_prior/config/ImitationLearning.yaml")
    use_expert_decoder = True
    if cfg_path.exists():
        net_cfg = yaml.safe_load(cfg_path.read_text())
        if net_cfg and "imitation_learning_policy" in net_cfg:
            use_expert_decoder = bool(net_cfg["imitation_learning_policy"].get("use_expert_decoder", True))
    else:
        print(f"[WARN] Config {cfg_path} not found. Defaulting use_expert_decoder=True")
        
    expert_decoder = expert.decoder if use_expert_decoder else None

    model = ImitationBlock(
        s_dim=s_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        expert_decoder=expert_decoder,
    ).to(device)

    loss_w = LossWeights(action=1.0, mm=0.1, reg=0.05, vq=1.0)

    trainer_cfg = TrainerCfg(
        device=str(device),
        use_amp=args.use_amp,
        ckpt_dir=f"{args.log_dir}/ckpts",
        mm_warmup_steps=50_000,
        log_every=50,
        ckpt_every=1000,
        mm_end=1.0,
        mm_start=0.1,
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
    
    if hasattr(env.unwrapped, "ref_frame_idx"):
        prev_ref_idx = env.unwrapped.ref_frame_idx.clone()
    else:
        raise RuntimeError("Env must expose env.unwrapped.ref_frame_idx")

    wrap_prev = torch.zeros_like(done_prev)

    # Loop
    for it in tqdm(range(args.steps), desc="Training Imitation"):
        obs_policy = obs["policy"]
        s, goal = _split_obs(obs_policy)
        
        valid_prev = (~done_prev) & (~wrap_prev)

        mu_expert = _get_mu_expert(expert, s, goal).to(device=device)

        obs, rew, terminated, truncated, extras = _step_env(env, mu_expert)
        done_after_step = (terminated | truncated).to(device=device, dtype=torch.bool)
        
        new_ref_idx = env.unwrapped.ref_frame_idx

        wrap_now = (new_ref_idx == 0) & (prev_ref_idx != 0) & (~done_after_step)
        valid_prev = (~done_prev) & (~wrap_now)

        if args.debug_temporal:
            if valid_prev.any():
                expected = prev_ref_idx + 1
                bad = valid_prev & (new_ref_idx != expected)
                if bad.any():
                    pass

        stats = trainer.train_step({"s": s, "goal": goal, "a_expert": mu_expert, "valid_prev": valid_prev})

        done_prev = done_after_step
        prev_ref_idx = new_ref_idx.clone()
        wrap_prev = wrap_now

        if (it + 1) % 1000 == 0:
            valid_ratio = valid_prev.float().mean().item()
            print(
                f"[it={it+1}] loss_total={stats['loss_total']:.4f} "
                f"loss_action={stats['loss_action']:.4f} loss_mm={stats['loss_mm']:.4f} "
                f"valid_prev={valid_ratio:.3f}"
            )

    print("[INFO] Closing environment...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()