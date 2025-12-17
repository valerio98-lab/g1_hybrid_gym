import torch
import gymnasium as gym

@torch.no_grad()
def main():
    env = gym.make("Template-G1-Hybrid-Gym-Direct-v0")
    obs, _ = env.reset()

    u = env.unwrapped
    frame0 = u.dataset[0]

    # joints
    jp = u.robot.data.joint_pos[:, u.dataset_to_isaac_indexes]
    err_j = torch.norm(jp - frame0["joints"].to(u.device), dim=-1).mean().item()

    # root pos
    rs = u.robot.data.root_state_w
    root_pos = rs[:, 0:3]
    root_quat = rs[:, 3:7]
    env_origins = u.scene.env_origins
    target_pos = frame0["root_pos"].to(u.device) + env_origins
    target_quat = frame0["root_quat_wxyz"].to(u.device)

    err_p = torch.norm(root_pos - target_pos, dim=-1).mean().item()

    # quat: compare q and -q
    e1 = torch.norm(root_quat - target_quat, dim=-1)
    e2 = torch.norm(root_quat + target_quat, dim=-1)
    err_q = torch.minimum(e1, e2).mean().item()

    print("=== RESET CONSISTENCY ===")
    print(f"mean joint_pos error: {err_j:.6e}")
    print(f"mean root_pos error:  {err_p:.6e}")
    print(f"mean root_quat error: {err_q:.6e}")

if __name__ == "__main__":
    main()
