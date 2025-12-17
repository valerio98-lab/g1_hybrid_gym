import torch
from pathlib import Path
from g1_hybrid_prior.dataset import G1HybridPriorDataset
from g1_hybrid_prior.helpers import (
    quat_rotate,
    quat_rotate_inv,
    quat_log,
    quat_inv,
    quat_normalize,
    quat_mul,
)

@torch.no_grad()
def main():
    ds = G1HybridPriorDataset(
        file_path=Path("/home/valerio/g1_hybrid_prior/data_raw/LAFAN1_Retargeting_Dataset/g1/dance1_subject2.csv"),
        robot="g1",
        lazy_load=False,
    )
    dt = 1.0 / ds.robot_cfg.fps
    N = len(ds)

    t = N // 3
    prev = ds[t - 1]
    cur = ds[t]
    nxt = ds[t + 1]

    # -------- Linear velocity frame check --------
    v_world_fd = (nxt["root_pos"] - prev["root_pos"]) / (2.0 * dt)
    v_stored = cur["root_lin_vel"]

    q_prev = quat_normalize(prev["root_quat_wxyz"])

    v_world_from_body = quat_rotate(q_prev, v_stored)
    err_if_stored_body = torch.norm(v_world_from_body - v_world_fd).item()
    err_if_stored_world = torch.norm(v_stored - v_world_fd).item()

    print("=== DATASET LINEAR VEL FRAME CHECK ===")
    print(f"t={t}, dt={dt}")
    print(f"||v_world_fd|| = {torch.norm(v_world_fd).item():.6f}")
    print(f"||v_stored||   = {torch.norm(v_stored).item():.6f}")
    print(f"error if stored is BODY  (R(q)*v_stored vs v_world_fd): {err_if_stored_body:.6e}")
    print(f"error if stored is WORLD (v_stored vs v_world_fd):      {err_if_stored_world:.6e}")

    # -------- Angular velocity consistency check --------
    q_prev = quat_normalize(prev["root_quat_wxyz"])
    q_next = quat_normalize(nxt["root_quat_wxyz"])

    q_rel = quat_mul(quat_inv(q_prev), q_next)
    q_rel = quat_normalize(q_rel)

    # prev->next spans 2*dt, match dataset central difference usage
    w_body_fd = quat_log(q_rel) / (dt)
    w_stored = cur["root_ang_vel"]

    err_w = torch.norm(w_stored - w_body_fd).item()

    print("\n=== DATASET ANGULAR VEL CONSISTENCY ===")
    print(f"error (w_stored vs quat_log(inv(q_prev)*q_next)/(2dt)): {err_w:.6e}")

if __name__ == "__main__":
    main()
