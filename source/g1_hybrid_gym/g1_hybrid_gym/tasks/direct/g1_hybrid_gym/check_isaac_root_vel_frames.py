import argparse
from isaaclab.app import AppLauncher

# --- launch Kit first (omni.* available) ---
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- safe imports after Kit launch ---
import torch

from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg import (
    G1HybridGymEnvCfg,
)
from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env import G1HybridGymEnv


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.to(dtype=x.dtype)
    denom = m.sum().clamp_min(1.0)
    return float((x * m).sum() / denom)


@torch.no_grad()
def main():
    cfg = G1HybridGymEnvCfg()
    cfg.scene.num_envs = 4
    cfg.robot_cfg.spawn.articulation_props.fix_root_link = False

    env = G1HybridGymEnv(cfg=cfg, render_mode=None)
    env.reset()

    dt = float(env.cfg.sim.dt * env.cfg.decimation)
    T = 300

    # We'll store POST-step states (i.e., after env.step())
    link_pos, link_quat, v_link_rep, w_link_rep = [], [], [], []
    com_pos, v_com_rep, w_com_rep = [], [], []
    v_rootstate_rep, w_rootstate_rep = [], []
    done_flags = []

    for _ in range(T):
        a = 0.25 * torch.randn((env.num_envs, env.cfg.action_space), device=env.device)
        obs, rew, term, trunc, info = env.step(a)

        done = (term | trunc).clone()  # (N,)
        done_flags.append(done)

        # LINK-consistent state (pose+vel of link origin)
        ls = env.robot.data.root_link_state_w  # (N,13)
        link_pos.append(ls[:, 0:3].clone())
        link_quat.append(ls[:, 3:7].clone())
        v_link_rep.append(ls[:, 7:10].clone())
        w_link_rep.append(ls[:, 10:13].clone())

        # COM-consistent state (pose+vel of COM)
        cs = env.robot.data.root_com_state_w  # (N,13)
        com_pos.append(cs[:, 0:3].clone())
        v_com_rep.append(cs[:, 7:10].clone())
        w_com_rep.append(cs[:, 10:13].clone())

        # MIXED state (link pose + COM velocity) -- this is what root_state_w is
        rs = env.robot.data.root_state_w
        v_rootstate_rep.append(rs[:, 7:10].clone())
        w_rootstate_rep.append(rs[:, 10:13].clone())

    # Stack time
    link_pos = torch.stack(link_pos, dim=0)  # (T,N,3)
    v_link_rep = torch.stack(v_link_rep, dim=0)  # (T,N,3)
    w_link_rep = torch.stack(w_link_rep, dim=0)  # (T,N,3)

    com_pos = torch.stack(com_pos, dim=0)  # (T,N,3)
    v_com_rep = torch.stack(v_com_rep, dim=0)  # (T,N,3)
    w_com_rep = torch.stack(w_com_rep, dim=0)  # (T,N,3)

    v_rootstate_rep = torch.stack(v_rootstate_rep, dim=0)  # (T,N,3)
    w_rootstate_rep = torch.stack(w_rootstate_rep, dim=0)  # (T,N,3)

    done_flags = torch.stack(done_flags, dim=0)  # (T,N) bool

    # ---------------------------------------------------------------------
    # IMPORTANT FIX: timestamp / alignment
    # Your logs showed v_link_rep is closer to POST-step velocity:
    #   (p[t] - p[t-1]) / dt  ~ v[t]
    # not (p[t+1]-p[t])/dt ~ v[t].
    # So we evaluate the main consistency with forward-diff aligned to v[t].
    # ---------------------------------------------------------------------

    # Forward difference: dp[t] = p[t] - p[t-1]  for t=1..T-1
    dp_link = link_pos[1:] - link_pos[:-1]  # (T-1,N,3)
    dp_com = com_pos[1:] - com_pos[:-1]  # (T-1,N,3)

    v_fd_link_fwd = dp_link / dt  # (T-1,N,3)
    v_fd_com_fwd = dp_com / dt  # (T-1,N,3)

    # Align reported velocities to POST-step (index t=1..T-1)
    vL_post = v_link_rep[1:]  # (T-1,N,3)
    wL_post = w_link_rep[1:]  # (T-1,N,3)
    vC_post = v_com_rep[1:]  # (T-1,N,3)
    wC_post = w_com_rep[1:]  # (T-1,N,3)
    vRS_post = v_rootstate_rep[1:]  # (T-1,N,3)
    wRS_post = w_rootstate_rep[1:]  # (T-1,N,3)

    # Valid mask for forward diffs: exclude transitions that involve a reset at t or t-1
    valid_fwd = ~(done_flags[1:] | done_flags[:-1])  # (T-1,N)

    # Kinematic relation in WORLD:
    # v_link = v_com + ω × (p_link - p_com)
    r_world_post = link_pos[1:] - com_pos[1:]  # (T-1,N,3) = pL - pC
    v_link_from_com_post = vC_post + torch.cross(wC_post, r_world_post, dim=-1)

    # Errors (forward/post)
    err_A = torch.norm(v_fd_link_fwd - vL_post, dim=-1)  # (T-1,N)
    err_B = torch.norm(v_fd_com_fwd - vC_post, dim=-1)  # (T-1,N)
    err_C = torch.norm(
        v_fd_link_fwd - vRS_post, dim=-1
    )  # (T-1,N) root_state_w lin vel is COM -> should be worse
    err_D = torch.norm(v_fd_link_fwd - v_link_from_com_post, dim=-1)  # (T-1,N)
    err_D2 = torch.norm(
        vL_post - v_link_from_com_post, dim=-1
    )  # (T-1,N) should be small if link/com buffers are consistent
    err_E = torch.norm(wL_post - wC_post, dim=-1)  # (T-1,N) should be ~0

    print("=== ISAAC ROOT VELOCITIES: CONSISTENCY CHECK (POST-step aligned) ===")
    print(f"dt_effective = {dt}")
    print(
        f"valid samples (t-1,t no reset) = {int(valid_fwd.sum().item())} / {valid_fwd.numel()}"
    )
    print("")
    print(
        f"[A] mean ||FD_fwd(link_pos_w)/dt - root_link_state_w.lin_vel||   = {masked_mean(err_A, valid_fwd):.6e}"
    )
    print(
        f"[B] mean ||FD_fwd(com_pos_w)/dt  - root_com_state_w.lin_vel||    = {masked_mean(err_B, valid_fwd):.6e}"
    )
    print(
        f"[C] mean ||FD_fwd(link_pos_w)/dt - root_state_w.lin_vel (COM!)|| = {masked_mean(err_C, valid_fwd):.6e}"
    )
    print(
        f"[D] mean ||FD_fwd(link_pos_w)/dt - (v_com + ω×(pL-pC))||         = {masked_mean(err_D, valid_fwd):.6e}"
    )
    print(
        f"[D2] mean ||v_link_rep - (v_com + ω×(pL-pC))||                   = {masked_mean(err_D2, valid_fwd):.6e}"
    )
    print(
        f"[E] mean ||ω_link - ω_com||                                      = {masked_mean(err_E, valid_fwd):.6e}"
    )
    print("")
    print("Interpretation:")
    print(
        "- [A] should be small if link velocities match link origin motion (with correct timestamp alignment)."
    )
    print("- [B] should be small if COM velocities match COM motion.")
    print("- [C] expected larger: root_state_w mixes LINK pose with COM velocity.")
    print("- [D] should be comparable to [A] if the rigid-body relation holds.")
    print(
        "- [D2] should be very small: internal consistency between link/com velocity buffers."
    )
    print("- [E] should be ~0 for a rigid body (and usually is).")
    # --- NEW: trapezoidal integration check ---
    # dp[t] ≈ 0.5 * (v[t-1] + v[t]) * dt
    vL_prev = v_link_rep[:-1]  # (T-1,N,3) velocity at t-1
    vL_curr = v_link_rep[1:]  # (T-1,N,3) velocity at t
    dp_pred_trap = 0.5 * (vL_prev + vL_curr) * dt  # (T-1,N,3)

    err_F = torch.norm(dp_link - dp_pred_trap, dim=-1)  # (T-1,N)

    print("\n=== EXTRA: INTEGRATION CHECK (trapezoid) ===")
    print(
        f"[F] mean ||(p[t]-p[t-1]) - 0.5*(v[t-1]+v[t])*dt|| = {masked_mean(err_F, valid_fwd):.6e}"
    )

    # ---------------------------------------------------------------------
    # Optional: also show CENTRAL difference (still useful, but alignment-sensitive)
    # Central diff centered at index t (uses t-1,t+1), compare to v[t].
    # Here v[t] is POST-step at same t.
    # ---------------------------------------------------------------------
    if T >= 3:
        v_fd_link_c = (link_pos[2:] - link_pos[:-2]) / (2.0 * dt)  # (T-2,N,3)
        v_fd_com_c = (com_pos[2:] - com_pos[:-2]) / (2.0 * dt)  # (T-2,N,3)

        vL_mid = v_link_rep[1:-1]  # (T-2,N,3)
        vC_mid = v_com_rep[1:-1]  # (T-2,N,3)

        valid_c = ~(done_flags[2:] | done_flags[1:-1] | done_flags[:-2])  # (T-2,N)

        err_Ac = torch.norm(v_fd_link_c - vL_mid, dim=-1)
        err_Bc = torch.norm(v_fd_com_c - vC_mid, dim=-1)

        print("\n=== OPTIONAL: CENTRAL-DIFF (centered) ===")
        print(
            f"valid samples (t-1,t,t+1 no reset) = {int(valid_c.sum().item())} / {valid_c.numel()}"
        )
        print(
            f"[Ac] mean ||FD_c(link_pos_w) - v_link_rep|| = {masked_mean(err_Ac, valid_c):.6e}"
        )
        print(
            f"[Bc] mean ||FD_c(com_pos_w)  - v_com_rep||  = {masked_mean(err_Bc, valid_c):.6e}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
