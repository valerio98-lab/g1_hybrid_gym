from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fk_sync.config import load_skeleton_configs
from fk_sync.kinematics import KinematicsModel
from fk_sync.tool import FkTool
from fk_sync.validation import validate_fk_against_sim

# IsaacLab bits (servono solo se fai validate contro sim)
from isaaclab.app import AppLauncher
import gymnasium as gym

from fk_sync.adapters.isaaclab import IsaacLabFKAdapter


DEEP_MIMIC_EE_DEFAULT = [
    # piedi
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    # mani (palmo)
    "left_hand_palm_link",
    "right_hand_palm_link",
    # se nel tuo URDF esistono e li vuoi aggiungere:
    # "head_link",
]


def main():
    parser = argparse.ArgumentParser("FK validate (IsaacLab) + optional CSV augment")

    parser.add_argument("--robots_yaml", type=str, required=True)
    parser.add_argument("--skeleton", type=str, default="g1")
    parser.add_argument("--urdf", type=str, required=True)

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV da cui campionare per validate e/o da augmentare",
    )
    parser.add_argument("--num_samples", type=int, default=200)

    parser.add_argument("--ee", type=str, nargs="*", default=DEEP_MIMIC_EE_DEFAULT)

    parser.add_argument(
        "--pos_thresh",
        type=float,
        default=1e-3,
        help="threshold posizione (m) per auto-select",
    )
    parser.add_argument(
        "--rot_thresh_deg",
        type=float,
        default=2.0,
        help="threshold orientazione (deg) per auto-select",
    )

    parser.add_argument(
        "--fk_source",
        type=str,
        choices=["auto", "internal", "simulator"],
        default="auto",
    )

    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="override output dir (altrimenti <csv_dir>_aug)",
    )

    # IsaacLab app args (headless/device etc.)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()

    # ---- Load skeleton config
    robots_yaml = Path(args.robots_yaml).expanduser().resolve()
    urdf_path = Path(args.urdf).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()

    skels = load_skeleton_configs(robots_yaml)
    skel = skels[args.skeleton]

    # ---- Tool (internal FK)
    tool = FkTool(
        dataset_yaml=robots_yaml,
        urdf_path=urdf_path,
        skeleton=args.skeleton,
        fk_source="internal",  # default, poi scegliamo sotto
    )

    # ---- Prepare q_samples from CSV (N rows)
    df = pd.read_csv(csv_path)
    # joint cols: assumiamo che nel CSV ci siano colonne con gli stessi nomi dei joint
    missing = [j for j in skel.joint_order if j not in df.columns]
    if missing:
        raise RuntimeError(
            "CSV non contiene tutte le colonne joint richieste.\n"
            "Mancano:\n" + "\n".join(f" - {m}" for m in missing)
        )

    N = min(args.num_samples, len(df))
    sample_df = df.sample(n=N, random_state=0) if len(df) > N else df
    q_samples = sample_df[skel.joint_order].to_numpy(dtype=float)  # (N, 29)

    # ---- If we need Isaac comparison (auto/simulator)
    need_sim = args.fk_source in ["auto", "simulator"]

    report = None
    if need_sim:
        # Launch Isaac app
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        # Crea un env qualsiasi che ti spawna il robot.
        # QUI devi passare il tuo task name. Se il tuo train usa `--task ...`, usa lo stesso.
        # Esempio: python scripts/fk_validate_and_augment.py ... --task g1_hybrid_gym:xxx
        if getattr(args, "task", None) is None:
            raise RuntimeError(
                "Per validate su IsaacLab devi passare anche --task (lo stesso che usi in training)."
            )

        env = gym.make(
            args.task, cfg=None
        )  # se usi hydra cfg nel tuo repo, sostituisci con la tua build cfg

        # robot articulation
        robot = env.unwrapped.robot  # nel tuo env c’è self.robot

        # adapter: facciamo uno step per aggiornare transforms
        step_fn = None
        if hasattr(env.unwrapped, "sim") and hasattr(env.unwrapped.sim, "step"):
            step_fn = lambda: env.unwrapped.sim.step()
        adapter = IsaacLabFKAdapter(
            articulation=robot,
            joint_names=skel.joint_order,
            env_id=0,
            step_fn=step_fn,
            quat_format="xyzw",  # se scopri che è già wxyz metti "wxyz"
        )

        # validate
        kin_model: KinematicsModel = tool._kin  # same internal model
        report = validate_fk_against_sim(
            kin_model=kin_model,
            simulator=adapter,
            q_samples=q_samples,
            link_names=list(args.ee),
        )
        print(report.pretty())

        # decision
        pos_bad = report.global_max_pos_err > float(args.pos_thresh)
        rot_thresh_rad = float(args.rot_thresh_deg) * np.pi / 180.0
        rot_bad = (report.global_max_rot_err_rad is not None) and (
            report.global_max_rot_err_rad > rot_thresh_rad
        )

        if args.fk_source == "auto":
            chosen = "simulator" if (pos_bad or rot_bad) else "internal"
        else:
            chosen = args.fk_source

        tool.set_fk_source(chosen)
        print(
            f"\n[FK] Selected source = {chosen} (pos_bad={pos_bad}, rot_bad={rot_bad})"
        )

        # close env + sim
        env.close()
        simulation_app.close()

    else:
        tool.set_fk_source(args.fk_source)

    # ---- Optional augmentation
    if args.augment:
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
        out_csv = tool.augment_csv(
            csv_path=csv_path,
            end_effectors=list(args.ee),
            add_orientation=True,
            out_dir=out_dir,
        )
        print(f"\n[AUG] Wrote augmented CSV: {out_csv}")


if __name__ == "__main__":
    main()
