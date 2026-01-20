import torch

from .g1_hybrid_gym_env_base import G1HybridGymEnvBase


class G1HybridGymEnvImitation(G1HybridGymEnvBase):
    """Imitation env: usa body_pos (FK) per early-termination paper-style.

    Note:
      - Usa lo stesso cfg (G1HybridGymEnvCfg).
      - Parametri extra li leggi da config_param.yaml via _read_dataset_params (come già fai in Base).
      - Se body_pos non è presente nel dataset, fa fallback su EE (se disponibile).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ---------- Body mapping (optional, depends on NPZ content) ----------
        self.body_names = None
        self.body_isaac_indices = None

        # Il tuo dataset NPZ loader salva i nomi body in dataset._body_names
        self.cfg_params = self._read_dataset_params()
        self.early_term_thr = float(
            self.cfg_params.get("early_termination_dist_threshold", 0.5))
        if hasattr(self.dataset, "get_body_names"):
            self.body_names = self.dataset.get_body_names()

        if self.body_names:
            all_body_names = self.robot.body_names
            body_isaac_indices = []
            for b in self.body_names:
                if b not in all_body_names:
                    raise ValueError(
                        f"[{self.__class__.__name__}] Body link '{b}' not found in Isaac articulation bodies!"
                    )
                body_isaac_indices.append(all_body_names.index(b))

            self.body_isaac_indices = torch.tensor(
                body_isaac_indices, device=self.device, dtype=torch.long
            )
            print(
                f"[{self.__class__.__name__}] Mapped BODY: {self.body_names} -> {self.body_isaac_indices.tolist()}"
            )
        default_exclude = {"left_ankle_roll_link", "right_ankle_roll_link"}

        # Option B: allow override from YAML (recommended)
        # e.g. in config_param.yaml: dataset_params: { early_term_exclude_bodies: [left_ankle_roll_link, ...] }
        exclude_names = set(self.cfg_params.get("early_term_exclude_bodies", [])) | default_exclude

        self._exclude_body_local_ids = None  # ids within *dataset* body list K
        if self.body_names:
            name_to_k = {n: i for i, n in enumerate(self.body_names)}
            excl = [name_to_k[n] for n in exclude_names if n in name_to_k]
            if len(excl) > 0:
                self._exclude_body_local_ids = torch.tensor(excl, device=self.device, dtype=torch.long)
                print(f"[{self.__class__.__name__}] Early-term EXCLUDE (dataset-body ids): {excl}")

    def _build_reference_tensors(self) -> None:
        """Estende la build base aggiungendo body_pos se presente nei frame.

        Nel Base attuale vedo che stacchi solo ee_pos:
        if "ee_pos" in frames[0]: self._ref_ee_pos = stack(...)
        ma poi _get_ref_batch prova a usare anche _ref_body_pos / _ref_body_rot.
        Quindi qui aggiungiamo body_pos davvero.
        """
        super()._build_reference_tensors()
        if getattr(self.dataset, "lazy_load", False):
            return
        frames = getattr(self.dataset, "dataset", None)
        if not frames:
            return

        # BODY positions from NPZ frames
        if frames and ("body_pos" in frames[0]):
            self._ref_body_pos = torch.stack([f["body_pos"] for f in frames], dim=0).to(
                self.device
            )

        # (opzionale in futuro) se un domani esporti body_rot:
        if frames and ("body_rot" in frames[0]):
            self._ref_body_rot = torch.stack([f["body_rot"] for f in frames], dim=0).to(
                self.device
            )

    def _get_dones(self):
        terminated, time_out = super()._get_dones()

        # ref (cache per step)
        ref = self._cached_ref_tensors
        if ref is None:
            self.ref_frame_idx.clamp_(0, self.max_frame_idx)
            ref = self._get_ref_batch(self.ref_frame_idx)
            self._cached_ref_tensors = ref

        # --- Paper-style early termination: max body distance > threshold ---
        # Threshold: lo prendiamo da config_param.yaml se presente, altrimenti 0.5 (come PPO).

        if (
            ref.get("body_pos") is not None
            and self.body_isaac_indices is not None
        ):
            curr_body_pos = self.robot.data.body_state_w[:, self.body_isaac_indices, 0:3]
            curr_body_pos_rel = curr_body_pos - self.scene.env_origins.unsqueeze(1)

            pos_dist = torch.linalg.norm(curr_body_pos_rel - ref["body_pos"], dim=-1)  # (N,K)

            # mask excluded bodies (feet/contact)
            if self._exclude_body_local_ids is not None:
                pos_dist.index_fill_(dim=1, index=self._exclude_body_local_ids, value=0.0)
            
            max_dist = pos_dist.max(dim=1).values

            body_term = max_dist > self.early_term_thr
            terminated = terminated | body_term

            # riuso i debug fields già esistenti nel Base/PPO per logging rapido
            self._dbg_ee_term = body_term
            self._dbg_maxdist = max_dist

            return terminated, time_out

        if (ref.get("ee_pos") is not None) and (self.ee_isaac_indices is not None):
            ee_pos_w = self.robot.data.body_state_w[:, self.ee_isaac_indices, 0:3]
            ee_pos_rel = ee_pos_w - self.scene.env_origins.unsqueeze(1)  # (N,E,3)
            max_dist = torch.linalg.norm(ee_pos_rel - ref["ee_pos"], dim=-1).max(dim=1).values

            ee_term = max_dist > self.early_term_thr
            terminated = terminated | ee_term
            self._dbg_ee_term = ee_term
            self._dbg_maxdist = max_dist

        return terminated, time_out
