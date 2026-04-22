from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import yaml
from tqdm import tqdm
from collections import OrderedDict



from isaaclab.app import AppLauncher

OBS_CLIP = 5.0
PARENT_DIR = Path(__file__).parents[2]


def main():
    import datetime
    parser = argparse.ArgumentParser("Train imitation block (online distillation)")

    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=110_000)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=f"./logs/imitation/{datetime.datetime.now().strftime('%d_%m_%Y_%H%M%S')}")
    parser.add_argument("--run_name", type=str, default="g1_hybrid_imitation")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--resume_from_ckpt", type=str, default=None, help="(Optional) path to student checkpoint to resume from")

    parser.add_argument("--expert_checkpoint", type=str, default=None,
                        help="Path to expert checkpoint if you load weights manually")
    parser.add_argument("--debug_temporal", action="store_true",
                        help="Enable extra temporal consistency checks")

    # Aggiunge argomenti per Isaac Sim
    AppLauncher.add_app_launcher_args(parser)
    args, _unknown = parser.parse_known_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    print("[INFO] Simulation App started. Importing tasks and models...")
    
    ### DO NOT MOVE THIS IMPORTS ###
    import g1_hybrid_gym.tasks  # noqa: F401
    
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
    from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_imitation import G1HybridGymEnvImitation
    from g1_hybrid_prior.models.expert_policy import ExpertPolicy
    from g1_hybrid_prior.models.hybrid_imitation_block import ImitationBlock
    from g1_hybrid_prior.trainers.imitation_trainer import ImitationTrainer, TrainerCfg, LossWeights
    
    ##############################

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    env_cfg = G1HybridGymEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)

    print(f"[INFO] Creating environment on {device}...")
    env = G1HybridGymEnvImitation(cfg=env_cfg, render_mode=None)
    obs = _reset_env(env)
    obs_policy = obs["policy"]
    
    try:
        action_dim = int(env.action_space.shape[0])
        # Controllo se è uno spazio vettorizzato (num_envs, action_dim)
        if hasattr(env, "single_action_space"):
             action_dim = int(env.single_action_space.shape[0])
    except:
         action_dim = int(env_cfg.action_space)
    

    CUR_OBS_DIM = env.CUR_OBS_DIM
    GOAL_DIM = env.GOAL_DIM
    print(f"[INFO] Dimensions: S={CUR_OBS_DIM}, Goal={GOAL_DIM}, Action={action_dim}")


    EXPERT_CKPT = args.expert_checkpoint
    if EXPERT_CKPT is None:
        raise RuntimeError("--expert_checkpoint is needed")

    expert = ExpertPolicy(obs_dim=CUR_OBS_DIM, goal_dim=GOAL_DIM, action_dim=action_dim, device=str(device)).to(device)
    expert.eval()
    # carica pesi + running_mean_std dal .pth rl_games
    expert.load_from_rlgames(EXPERT_CKPT, strict=False, load_rms=True, enable_rms=False, clip=OBS_CLIP)
    
    if args.resume_from_ckpt is not None:
        resume_path = Path(args.resume_from_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume not found: {resume_path}")
        
        inferred_run_name = resume_path.parent.name
        inferred_log_dir = str(resume_path.parent.parent.parent)

        print(f"[INFO] Resume detected. Overriding:")
        print(f"log_dir = {inferred_log_dir}")
        print(f" run_name = {inferred_run_name}")

        args.log_dir = inferred_log_dir
        args.run_name = inferred_run_name
    
    cfg_path = Path(PARENT_DIR / "train_config/ImitationLearning.yaml")
    config_param_path = PARENT_DIR.parents[2] / "source/g1_hybrid_gym/config/config_param.yaml"
    
    use_expert_decoder = False
    if cfg_path.exists():
        net_cfg = yaml.safe_load(cfg_path.read_text())
        if net_cfg and "imitation_learning_policy" in net_cfg:
            os.makedirs(f"{args.log_dir}_{args.experiment_name}", exist_ok=True)

            with open(f"{args.log_dir}_{args.experiment_name}/imitation_params.yaml", "w") as f: 
                yaml.dump(net_cfg, f)
            
            if config_param_path.exists():
                config_param = yaml.safe_load(config_param_path.read_text())
                with open(f"{args.log_dir}_{args.experiment_name}/config_param.yaml", "w") as f:
                    yaml.dump(config_param, f)

            net_cfg = net_cfg["imitation_learning_policy"]
            use_expert_decoder = bool(net_cfg.get("use_expert_decoder", False))
    else:
        print(f"[WARN] Config {cfg_path} not found. Defaulting use_expert_decoder=True")
        
    expert_decoder = expert.decoder if use_expert_decoder else None

    model = ImitationBlock(
        s_dim=CUR_OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=action_dim,
        expert_decoder=expert_decoder,
        net_cfg_path=cfg_path
    ).to(device)

    loss_w = LossWeights(
            action=net_cfg["loss_weights"]["action"], 
            mm=net_cfg["loss_weights"]["mm"], 
            reg=net_cfg["loss_weights"]["reg"], 
            vq=net_cfg["loss_weights"]["vq"]
        )

    trainer_cfg = TrainerCfg(
        device=str(device),
        use_amp=net_cfg["trainer_cfg"]["use_amp"],
        ckpt_dir=f"{args.log_dir}_{args.experiment_name}/ckpts",
        mm_warmup_steps=net_cfg["trainer_cfg"]["mm_warmup_steps"],
        log_every=net_cfg["trainer_cfg"]["log_every"],
        ckpt_every=net_cfg["trainer_cfg"]["ckpt_every"],
        mm_end=net_cfg["trainer_cfg"]["mm_end"],
        mm_start=net_cfg["trainer_cfg"]["mm_start"],
        lr=net_cfg["trainer_cfg"]["lr"],
        keep_last_k=net_cfg["trainer_cfg"]["keep_last_k"],
    )

    trainer = ImitationTrainer(
        model=model,
        loss_weights=loss_w,
        cfg=trainer_cfg,
        log_dir=args.log_dir,
        run_name=args.run_name,
    )

    if args.resume_from_ckpt is not None:
        print(f"[INFO] Resuming student from {args.resume_from_ckpt}")
        trainer.load(args.resume_from_ckpt)
        print(f"[INFO] Resumed at global_step={trainer.global_step}")

    obs = _reset_env(env)
    done_prev = torch.ones((env.unwrapped.num_envs,), device=device, dtype=torch.bool)
    
    if hasattr(env.unwrapped, "ref_frame_idx"):
        prev_ref_idx = env.unwrapped.ref_frame_idx.clone()
    else:
        raise RuntimeError("Env must expose env.unwrapped.ref_frame_idx")

    wrap_prev = torch.zeros_like(done_prev)

    # Loop
    for _ in tqdm(range(args.steps), desc="Training Imitation"):
        obs_policy = obs["policy"]
        s, goal = _split_obs(obs_policy, CUR_OBS_DIM=CUR_OBS_DIM)

        full = torch.cat([s, goal], dim=-1)                            # (N, 332)
        full_n = expert.obs_rms.normalize(full, clip=OBS_CLIP)         # (N, 332)
        s_n = full_n[..., :CUR_OBS_DIM]
        goal_n = full_n[..., CUR_OBS_DIM:]

        #print(f"[TrainImitation]: STATE_DIM: {s_n.shape[-1]}, GOAL_DIM: {goal_n.shape[-1]}")
                
        # --- valid_prev refers to (t-1)->t, so use wrap_prev (wrap happened last step) ---
        valid_prev = (~done_prev) & (~wrap_prev)

        mu_expert = _get_mu_expert(expert, s_n, goal_n).to(device=device)

        ####### DEBUG: Sanity check on action-magnitude #######
        mse_zero = (mu_expert ** 2).mean()  # scalar
        # optional: action RMS / abs mean to understand scale
        act_rms = mu_expert.pow(2).mean().sqrt()
        act_abs = mu_expert.abs().mean()
        ##########################################

        obs, rew, terminated, truncated, extras = _step_env(env, mu_expert)
        done_after_step = (terminated | truncated).to(device=device, dtype=torch.bool)

        new_ref_idx = env.unwrapped.ref_frame_idx

        max_idx = env.unwrapped.max_frame_idx
        wrap_now = (prev_ref_idx == max_idx) & (new_ref_idx == 0) & (~done_after_step)

        if args.debug_temporal:
            expected_new = prev_ref_idx + 1
            expected_new = torch.where(prev_ref_idx == max_idx, torch.zeros_like(expected_new), expected_new)
            bad = (~done_after_step) & (new_ref_idx != expected_new)
            if bad.any():
                idx = bad.nonzero(as_tuple=False).squeeze(-1)
                print(f"[WARN] Temporal inconsistency at step={trainer.global_step}: count={idx.numel()} first={idx[:20].tolist()}")
                                
        stats = trainer.train_step({"s": s_n, "goal": goal_n, "a_expert": mu_expert, "valid_prev": valid_prev})

        done_prev = done_after_step
        prev_ref_idx = new_ref_idx.clone()
        wrap_prev = wrap_now

        if (trainer.global_step % 1000) == 0:
            valid_ratio = valid_prev.float().mean().item()
            nmse = stats["loss_action"] / (mse_zero + 1e-8)

            print(
                f"[step={trainer.global_step}] loss_total={stats['loss_total']:.4f} "
                f"loss_action={stats['loss_action']:.4f} loss_mm={stats['loss_mm']:.4f} loss_commit={stats['loss_commit']:.4f} loss_reg={stats['loss_reg']:.4f}\n"
                f"mse_zero={mse_zero:.6f} act_rms={act_rms:.6f} act_abs={act_abs:.6f} "
                f"valid_prev={valid_ratio:.3f} nmse={nmse:.4f}"
            )
            with torch.no_grad():
                a_hat = trainer.model(s_n, goal_n)["a_hat"].detach()

                # (A) quanto è "non-zero" a_hat
                mse_a_hat_zero = (a_hat ** 2).mean().item()                 # MSE(a_hat, 0)
                rms_a_hat = a_hat.pow(2).mean().sqrt().item()
                abs_a_hat = a_hat.abs().mean().item()

                # (B) allineamento direzionale con expert (cos-sim medio)
                eps = 1e-8
                cos = torch.nn.functional.cosine_similarity(a_hat, mu_expert, dim=-1, eps=eps)
                cos_mean = cos.mean().item()

            print(f"mse_hat_zero={mse_a_hat_zero:.6f} hat_rms={rms_a_hat:.6f} hat_abs={abs_a_hat:.6f} cos_hat_exp={cos_mean:.4f}")

    print("[INFO] Closing environment...")
    env.close()
    simulation_app.close()


def _reset_env(env) -> Dict[str, torch.Tensor]:
    # Gymnasium compat
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
        return obs
    return out

def _step_env(env, action: torch.Tensor):
    return env.step(action)

def _split_obs(obs_policy: torch.Tensor, CUR_OBS_DIM) -> Tuple[torch.Tensor, torch.Tensor]:
    s = obs_policy[..., :CUR_OBS_DIM]
    goal = obs_policy[..., CUR_OBS_DIM:]
    return s, goal

@torch.no_grad()
def _get_mu_expert(expert: Any, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    mu, _, _ = expert(s, goal)
    return mu


if __name__ == "__main__":
    main()