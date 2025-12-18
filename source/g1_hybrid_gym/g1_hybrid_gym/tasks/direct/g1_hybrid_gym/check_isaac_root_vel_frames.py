import argparse
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# 1. LAUNCH KIT (Must be done before any torch/isaaclab imports)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Test Velocity Matching: Dataset vs Isaac Observation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headless if you don't need UI, or remove to see the robot
# args.headless = True 

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# 2. IMPORTS (Safe after Kit launch)
# -----------------------------------------------------------------------------
import torch
import numpy as np

# Importa le classi dal tuo Environment e Dataset
from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env_cfg import G1HybridGymEnvCfg
from g1_hybrid_gym.tasks.direct.g1_hybrid_gym.g1_hybrid_gym_env import G1HybridGymEnv

from g1_hybrid_prior.helpers import (
    quat_rotate,
    quat_normalize,
)

def main():
    print("[Test] Initializing Environment...")
    
    # --- Configurazione ---
    cfg = G1HybridGymEnvCfg()
    cfg.scene.num_envs = 1      # Testiamo su un solo environment per chiarezza
    cfg.sim.dt = 1.0 / 30.0     # Forziamo il DT a matchare i 30FPS del dataset
    cfg.decimation = 1          # Decimation 1 per controllo diretto step-by-step
    
    # Disabilita reset automatici per altezza, vogliamo controllo manuale
    cfg.min_height_reset = -100.0 

    # --- Creazione Env ---
    env = G1HybridGymEnv(cfg=cfg, render_mode=None)
    
    # Primo reset per inizializzare i buffer interni
    env.reset()
    
    # --- Selezione Frame di Test ---
    # Prendiamo un frame centrale dove le velocità "central" sono ben definite
    frame_idx = 150 
    print(f"[Test] Loading Frame {frame_idx} from Dataset...")
    
    # Accesso diretto al dataset caricato dall'env
    data = env.dataset[frame_idx]
    
    # Recuperiamo i target dal dataset
    # NOTA: Grazie alla tua modifica "central", queste velocità sono già 
    # proiettate nel BODY FRAME all'istante t (Corrente)
    v_ref_body = data["root_lin_vel"].to(env.device).unsqueeze(0) # (1, 3)
    w_ref_body = data["root_ang_vel"].to(env.device).unsqueeze(0) # (1, 3)
    
    root_pos = data["root_pos"].to(env.device).unsqueeze(0)       # (1, 3)
    root_quat = data["root_quat_wxyz"].to(env.device).unsqueeze(0)# (1, 4)

    # -------------------------------------------------------------------------
    # 3. PREPARAZIONE STATO PER ISAAC (Conversione Body -> World -> COM)
    # -------------------------------------------------------------------------
    # Isaac vuole posizione e velocità in WORLD coordinates.
    # Inoltre, PhysX gestisce la velocità del COM (Center of Mass), non del Link geometrico.
    
    # A. Body -> World
    v_link_world = quat_rotate(root_quat, v_ref_body)
    w_link_world = quat_rotate(root_quat, w_ref_body)
    
    # B. Link -> COM (PhysX logic)
    # v_com = v_link + w x (r_com - r_link)
    # r_body è il vettore (COM - Link) nel frame locale
    r_body = env.robot.data.body_com_pos_b[0, 0].unsqueeze(0) # (1, 3)
    r_world = quat_rotate(root_quat, r_body)     # (1, 3)
    
    v_com_world = v_link_world + torch.linalg.cross(w_link_world, r_world, dim=-1)
    
    # Costruiamo il tensore di stato completo
    root_state = torch.zeros((1, 13), device=env.device)
    root_state[:, 0:3] = root_pos + env.scene.env_origins # Absolute position
    root_state[:, 3:7] = root_quat
    root_state[:, 7:10] = v_com_world
    root_state[:, 10:13] = w_link_world # Angular vel is roughly same for rigid body

    print("[Test] Writing State to Sim...")
    
    # -------------------------------------------------------------------------
    # 4. SCRITTURA E AGGIORNAMENTO BUFFER
    # -------------------------------------------------------------------------
    # Scriviamo direttamente nella memoria di PhysX
    env.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=torch.tensor([0], device=env.device))
    env.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=torch.tensor([0], device=env.device))
    
    # CRUCIALE: Dobbiamo forzare l'aggiornamento delle "viste" di IsaacLab (env.robot.data)
    # senza fare un passo di fisica (che integrerebbe la posizione cambiandola).
    # write_data_to_sim flusha i dati verso PhysX
    env.scene.write_data_to_sim()
    
    # update_buffers rilegge i dati da PhysX e popola env.robot.data
    # dt piccolo serve solo per i calcoli interni, qui non avanziamo la fisica
    env.robot.update(dt=0.001) 

    # -------------------------------------------------------------------------
    # 5. VERIFICA OSSERVAZIONE (Il vero test)
    # -------------------------------------------------------------------------
    print("[Test] Computing Observations...")
    
    # Chiamiamo la tua funzione _get_observations che fa la proiezione inversa World->Body
    obs_dict = env._get_observations()
    obs = obs_dict["policy"] # Tensore [h, q, v, w, ...]
    
    # Estraiamo v e w dal tensore di osservazione.
    # Layout atteso da G1HybridGymEnv:
    # [h(1), q(4), v_body(3), w_body(3), ...]
    # Indici:
    # h: 0
    # q: 1:5
    # v_body: 5:8
    # w_body: 8:11
    
    v_isaac_body = obs[0, 5:8]
    w_isaac_body = obs[0, 8:11]
    
    # --- RISULTATI ---
    print("\n" + "="*50)
    print("CONFRONTO VELOCITÀ: DATASET (Target) vs ISAAC (Obs)")
    print("="*50)
    
    v_ref_np = v_ref_body.cpu().numpy()[0]
    v_sim_np = v_isaac_body.cpu().numpy()
    
    print(f"Dataset Linear Vel (Body):  [{v_ref_np[0]:.5f}, {v_ref_np[1]:.5f}, {v_ref_np[2]:.5f}]")
    print(f"Isaac   Linear Vel (Body):  [{v_sim_np[0]:.5f}, {v_sim_np[1]:.5f}, {v_sim_np[2]:.5f}]")
    
    diff_v = torch.norm(v_ref_body - v_isaac_body).item()
    print(f"-> Linear Velocity Error (L2): {diff_v:.8f}")
    
    print("-" * 30)
    
    w_ref_np = w_ref_body.cpu().numpy()[0]
    w_sim_np = w_isaac_body.cpu().numpy()
    
    print(f"Dataset Angular Vel (Body): [{w_ref_np[0]:.5f}, {w_ref_np[1]:.5f}, {w_ref_np[2]:.5f}]")
    print(f"Isaac   Angular Vel (Body): [{w_sim_np[0]:.5f}, {w_sim_np[1]:.5f}, {w_sim_np[2]:.5f}]")
    
    diff_w = torch.norm(w_ref_body - w_isaac_body).item()
    print(f"-> Angular Velocity Error (L2): {diff_w:.8f}")
    
    print("="*50)
    
    if diff_v < 1e-4:
        print("\n✅ SUCCESS: Le velocità matchano perfettamente!")
        print("Il sistema di coordinate (Dataset -> World -> Isaac -> Body) è COERENTE.")
    else:
        print("\n❌ FAILURE: C'è una discrepanza significativa.")
        print("Possibili cause:")
        print("1. Dataset usa proiezione su t-1 mentre Env su t (o viceversa).")
        print("2. Errore nella conversione Link <-> Center of Mass.")
        print("3. I quaternioni sono normalizzati male.")

    # Chiude l'app
    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        simulation_app.close()