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
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import wrap_env

from g1_hybrid_gym.envs.g1_hybrid_gym_env_amp import G1HybridGymEnvAMP
from g1_hybrid_gym.envs.g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
from g1_hybrid_gym.skrl_utils import PolicyWrapperAMP, PolicyValueWrapper
from g1_hybrid_prior.expert_policy import LowLevelExpertPolicy


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

    expert_net = LowLevelExpertPolicy(
        obs_dim=state_dim, goal_dim=state_dim, action_dim=action_dim, device=device
    )

    # Wrappers (Discriminatore non serve in play, ma PPO lo richiede nella config se AMP=True)
    # Possiamo passargli None o un dummy se SKRL lo permette, ma per sicurezza ricreiamo la struttura
    policy = PolicyWrapperAMP(
        env.observation_space, env.action_space, device, expert_net
    )
    value = PolicyValueWrapper(
        env.observation_space, env.action_space, device, expert_net
    )

    # Config minima per caricare
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["algorithm"]["amp"] = True

    agent = PPO(
        models={"policy": policy, "value": value},
        memory=None,
        cfg=cfg_ppo,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Carica Pesi
    print(f"[Play] Loading checkpoint: {args_cli.checkpoint}")
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval")

    # Loop
    obs, _ = env.reset()

    print("[Play] Visualizing...")
    while simulation_app.is_running():
        with torch.no_grad():
            # act restituisce le azioni
            actions = agent.act(obs, timestep=0, timesteps=0)[0]

        obs, reward, terminated, truncated, info = env.step(actions)

    simulation_app.close()


if __name__ == "__main__":
    main()
