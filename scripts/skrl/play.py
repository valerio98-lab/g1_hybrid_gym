# play_amp.py
import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play AMP Agent")
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to .pt checkpoint"
)
parser.add_argument("--num_envs", type=int, default=64, help="Envs to visualize")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Forza grafica
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# --- MODIFICA 1: Import AMP e Wrapper Corretto ---
from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env

# -------------------------------------------------

import g1_hybrid_gym.tasks  # noqa: F401
from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.rl.wrapper_amp import (
    PolicyWrapperAMP,
    PolicyValueWrapper,
)

from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_amp import (
    G1HybridGymEnvAMP,
)

from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg import (
    G1HybridGymEnvCfg,
)

# --- MODIFICA 2: Import Classi Separate ---
from g1_hybrid_prior.expert_policy import LowLevelActor, LowLevelCritic


def main():
    # Env Config
    env_cfg = G1HybridGymEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.sim.device = device

    # Init Env
    print("[Play] Starting Environment...")
    env = G1HybridGymEnvAMP(cfg=env_cfg, render_mode=None)
    env = wrap_env(env, wrapper="isaaclab")

    # Ricostruzione Modello (Identica al Train)
    full_obs_dim = env.observation_space.shape[0]
    state_dim = full_obs_dim // 2
    action_dim = env.action_space.shape[0]

    # --- MODIFICA 3: Istanziazione Reti Separate ---
    print("[Play] Rebuilding Actor and Critic Networks...")
    actor_net = LowLevelActor(
        obs_dim=state_dim, goal_dim=state_dim, action_dim=action_dim, device=device
    )
    critic_net = LowLevelCritic(obs_dim=state_dim, goal_dim=state_dim, device=device)

    # Wrappers (Colleghiamo le reti specifiche)
    policy = PolicyWrapperAMP(
        env.observation_space, env.action_space, device, actor_net
    )
    value = PolicyValueWrapper(
        env.observation_space, env.action_space, device, critic_net
    )

    # Non istanziamo il discriminatore qui, tanto non dobbiamo addestrarlo.
    models = {"policy": policy, "value": value}

    # --- Configurazione AMP ---
    cfg_amp = AMP_DEFAULT_CONFIG.copy()
    cfg_amp["experiment"]["write_interval"] = 0  # Niente log
    cfg_amp["experiment"]["checkpoint_interval"] = 0  # Niente checkpoint

    # --- Istanziazione Agente AMP ---
    # Non passiamo memory, motion_dataset, reply_buffer né loader perché in Eval non servono.
    agent = AMP(
        models=models,
        memory=None,
        cfg=cfg_amp,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Inizializzazione interna SKRL
    agent.init()

    # Carica Pesi
    print(f"[Play] Loading checkpoint: {args_cli.checkpoint}")
    agent.load(args_cli.checkpoint)

    # IMPORTANTE: Mettere in modalità valutazione (disattiva dropout, noise, update, ecc.)
    agent.set_running_mode("eval")

    # Loop
    obs, _ = env.reset()

    print("[Play] Visualizing...")
    while simulation_app.is_running():
        with torch.no_grad():
            # act restituisce le azioni
            # SKRL act ritorna: actions, log_prob, outputs
            actions = agent.act(obs, timestep=0, timesteps=0)[0]

        obs, reward, terminated, truncated, info = env.step(actions)

    simulation_app.close()


if __name__ == "__main__":
    main()
