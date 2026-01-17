# train_amp.py
import argparse
import sys
import os
from datetime import datetime

# --- 1. ISAAC LAB APP LAUNCHER ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train AMP Expert Policy with SKRL")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record video during training"
)
parser.add_argument(
    "--num_envs", type=int, default=4096, help="Number of parallel environments"
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to checkpoint to resume training"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# SKRL IMPORTS
from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# LOCAL IMPORTS
import g1_hybrid_gym.tasks  # noqa: F401
from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.rl.wrapper_amp import (
    PolicyWrapperAMP,
    PolicyValueWrapper,
    AMPDiscriminator,
)

from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_amp import (
    G1HybridGymEnvAMP,
)

from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg import (
    G1HybridGymEnvCfg,
)
from g1_hybrid_prior.expert_policy import LowLevelActor, LowLevelCritic
import gymnasium as gym


class AMPLoggingWrapper(gym.Wrapper):
    def __init__(self, env, discriminator, cfg_amp: dict, device: str):
        super().__init__(env)
        self.discriminator = discriminator
        self.device = torch.device(device)
        self.device_type = self.device.type

        self.w_task = float(cfg_amp.get("task_reward_weight", 0.5))
        self.w_style = float(cfg_amp.get("style_reward_weight", 0.5))
        self.disc_reward_scale = float(cfg_amp.get("discriminator_reward_scale", 2.0))

        # episode accumulators (per-env)
        n = self.env.num_envs
        self._ep_task = torch.zeros(n, device=self.device)
        self._ep_style = torch.zeros(n, device=self.device)
        self._ep_combined = torch.zeros(n, device=self.device)

    @property
    def num_envs(self):
        return self.env.num_envs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # safe reset accumulators
        self._ep_task.zero_()
        self._ep_style.zero_()
        self._ep_combined.zero_()
        return obs, info

    @torch.no_grad()
    def step(self, action):
        obs, rew, terminated, truncated, infos = self.env.step(action)

        if infos is None:
            infos = {}
        if isinstance(infos, list):
            infos = {}

        amp_obs = infos.get("amp_obs", None)  # (N, amp_dim)
        if amp_obs is None:
            return obs, rew, terminated, truncated, infos
        amp_obs = amp_obs.to(self.device)
        with torch.autocast(device_type=self.device_type, enabled=False):
            logits, _, _ = self.discriminator.act(
                {"states": amp_obs}, role="discriminator"
            )
            # prob = 1 / (1 + exp(-logits))
            prob = 1.0 / (1.0 + torch.exp(-logits))
            # style = -log(max(1 - prob, 1e-4))
            style = -torch.log(
                torch.maximum(1.0 - prob, torch.tensor(1e-4, device=self.device))
            )
            style = style.view_as(rew) * self.disc_reward_scale  # (N,)

        combined = self.w_task * rew + self.w_style * style  # (N,)

        # accumula per episodio
        self._ep_task += rew
        self._ep_style += style
        self._ep_combined += combined

        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)

            log = infos.setdefault("log", {})
            log["Reward / task_ep_mean"] = float(self._ep_task[done_ids].mean().item())
            log["Reward / style_ep_mean"] = float(
                self._ep_style[done_ids].mean().item()
            )
            log["Reward / combined_ep_mean"] = float(
                self._ep_combined[done_ids].mean().item()
            )

            # reset accumulators per gli env che hanno finito episodio
            self._ep_task[done_ids] = 0.0
            self._ep_style[done_ids] = 0.0
            self._ep_combined[done_ids] = 0.0

        return obs, rew, terminated, truncated, infos


def main():
    set_seed(42)

    env_cfg = G1HybridGymEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.sim.device = device

    print(f"[Train] Initializing G1HybridGymEnvAMP on {device}...")
    env_custom = G1HybridGymEnvAMP(
        cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    env = wrap_env(env_custom, wrapper="isaaclab")

    full_obs_dim = env.observation_space.shape[0]
    state_dim = full_obs_dim // 2
    action_dim = env.action_space.shape[0]

    print(f"[Train] Obs: {full_obs_dim} | State: {state_dim} | Action: {action_dim}")

    actor_net = LowLevelActor(
        obs_dim=state_dim, goal_dim=state_dim, action_dim=action_dim, device=device
    )
    critic_net = LowLevelCritic(obs_dim=state_dim, goal_dim=state_dim, device=device)

    policy = PolicyWrapperAMP(
        env.observation_space, env.action_space, device, actor_net
    )
    value = PolicyValueWrapper(
        env.observation_space, env.action_space, device, critic_net
    )

    def unwrap_to_custom(env):
        curr = env
        while hasattr(curr, "env"):
            curr = curr.env
        return curr

    custom_env = unwrap_to_custom(env)

    print("AMP obs dim (env):", custom_env.amp_observation_space.shape[0])
    print("Demo sample dim:", custom_env.fetch_amp_expert_batch(8).shape)

    amp_dim = custom_env.amp_observation_space.shape[0]
    print(f"[Train] AMP Obs Dim (Discriminator Input): {amp_dim}")
    if amp_dim != state_dim * 2:
        print(
            f"⚠️ WARNING: AMP Dim ({amp_dim}) != 2 * State Dim ({state_dim*2}). "
            "Assicurati che num_amp_obs_steps=2 in config_param.yaml se vuoi seguire il paper."
        )

    discriminator = AMPDiscriminator(
        env.observation_space, env.action_space, device, input_dim=amp_dim
    )

    models = {"policy": policy, "value": value, "discriminator": discriminator}

    cfg_amp = AMP_DEFAULT_CONFIG.copy()

    cfg_amp["learning_rate"] = 5e-5
    cfg_amp["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_amp["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

    cfg_amp["rollouts"] = 16
    cfg_amp["learning_epochs"] = 6
    cfg_amp["mini_batches"] = 1

    cfg_amp["discount_factor"] = 0.95
    cfg_amp["lambda"] = 0.95

    cfg_amp["ratio_clip"] = 0.1
    cfg_amp["value_clip"] = 0.1
    cfg_amp["clip_predicted_values"] = True
    cfg_amp["entropy_loss_scale"] = 0.0
    cfg_amp["grad_norm_clip"] = 1.0

    cfg_amp["amp_batch_size"] = 1024

    cfg_amp["task_reward_weight"] = 0.9
    cfg_amp["style_reward_weight"] = 0.1

    cfg_amp["discriminator_loss_scale"] = 5.0
    cfg_amp["discriminator_batch_size"] = 8192
    cfg_amp["discriminator_reward_scale"] = 2
    cfg_amp["discriminator_gradient_penalty_scale"] = 5
    cfg_amp["discriminator_logit_regularization_scale"] = 0.05
    cfg_amp["discriminator_weight_decay_scale"] = 1e-4

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_G1_AMP"
    cfg_amp["experiment"]["directory"] = os.path.join("logs", "skrl", run_name)
    cfg_amp["experiment"]["write_interval"] = 100
    cfg_amp["experiment"]["checkpoint_interval"] = 2000

    # env = AMPLoggingWrapper(env, discriminator, cfg_amp, device)

    # ------------------------------------------------------------------------
    # Funzione Loader
    # Questa funzione viene chiamata dall'agente dentro __init__ e durante il training
    # per pescare batch casuali dal tensore statico.
    def amp_motion_loader(n_samples: int):
        return custom_env.fetch_amp_expert_batch(n_samples)

    demo = amp_motion_loader(8)
    print("[Train] Demo sample dim:", demo.shape, flush=True)  # deve essere (8, K*69)

    # Memoria principale (Rollouts)
    memory = RandomMemory(
        memory_size=cfg_amp["rollouts"], num_envs=env.num_envs, device=device
    )

    # Motion Dataset (Dati Esperto) - Deve essere grande per contenere il dataset
    motion_dataset = RandomMemory(
        memory_size=200000, num_envs=1, device=device, replacement=False
    )

    # Reply Buffer (Dati Discriminatore) - Serve a stabilizzare il GAN
    reply_buffer = RandomMemory(
        memory_size=1000000, num_envs=1, device=device, replacement=False
    )

    print("[Train] Instantiating AMP Agent with Loader Function...")

    agent = AMP(
        models=models,
        memory=memory,
        cfg=cfg_amp,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        amp_observation_space=custom_env.amp_observation_space,  # Dimensione dello stato AMP (69)
        motion_dataset=motion_dataset,
        reply_buffer=reply_buffer,  # Memoria vuota per replay
        collect_reference_motions=amp_motion_loader,
    )
    if args_cli.checkpoint:
        print(f"[Train] Resuming from checkpoint: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)

    print("Starting Training...")
    trainer_cfg = {
        "timesteps": 500_000,
        "headless": True,
        "disable_progressbar": False,
        "close_environment_at_exit": True,
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()

    simulation_app.close()


if __name__ == "__main__":
    main()
