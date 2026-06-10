from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import (
    BATCH_SIZE, DEVICE, USE_CUDA,
    N_PRETRAIN_SAMPLES, N_SAMPLES, N_NEIGHBORS,
    PRETRAIN_EP, IP_EP, SOT_EP, VMF_EPOCHS, JOINT_EPOCHS,
    SA_EPOCHS, SA_RESTARTS,
    LR, WEIGHT_DECAY, NUM_WORKERS, PIN_MEMORY,
    KAPPA_MIN, KAPPA_MAX,
    CHECKPOINT_DIR, RESULTS_JSON_TPL,
)
from data.dataset       import MPIIFaceGazeDataset
from gpm.robust_gpm     import RobustGPM
from models.backbone    import ResNet18FeatureExtractor
from models.isometric_propagator import IsometricPropagator
from models.vmf_head    import vMFHead
from neuropalsy.dataset import NeuropalsyDataset, neuropalsy_collate
from utils.geometry     import angular_error_np, cone_95
from utils.metrics      import MetricLogger
from utils.device       import gpu_mem


class DualAGGFramework:

    def __init__(
        self,
        condition: str   = "nystagmus",
        severity:  float = 0.6,
    ) -> None:
        self.device    = DEVICE
        self.use_amp   = USE_CUDA
        self.scaler    = torch.amp.GradScaler("cuda") if USE_CUDA else None
        self.condition = condition
        self.severity  = severity

        # Model components
        self.cnn       = ResNet18FeatureExtractor(pretrained=True).to(self.device)
        self.fc        = nn.Linear(512, 3).to(self.device)
        self.ip:       IsometricPropagator | None = None
        self.gpm:      RobustGPM | None           = None
        self.gpm_patho: RobustGPM | None          = None
        self.vmf_head  = vMFHead().to(self.device)

        self._best_pretrain = float("inf")

        print(
            f"\n[DualAGG] condition={condition}  severity={severity}  "
            f"device={DEVICE}  batch={BATCH_SIZE}\n"
            f"          SA_RESTARTS={SA_RESTARTS}  SA_EPOCHS={SA_EPOCHS}  "
            f"IP_EP={IP_EP}  κ∈[{KAPPA_MIN},{KAPPA_MAX}]"
        )

    def _make_loader(self, ds, shuffle: bool, collate_fn=None) -> DataLoader:
        nw = NUM_WORKERS if os.name != "nt" else 0
        kw = dict(
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=PIN_MEMORY,
            persistent_workers=(nw > 0),
        )
        if USE_CUDA and nw > 0:
            kw["prefetch_factor"] = 2
        if collate_fn:
            kw["collate_fn"] = collate_fn
        return DataLoader(ds, **kw)

    def _fwd(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = imgs.to(self.device, non_blocking=True)
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                return self.cnn(imgs)
        return self.cnn(imgs)

    def _step(
        self,
        loss:        torch.Tensor,
        opt:         torch.optim.Optimizer,
        clip_params: list | None = None,
    ) -> None:
        if self.use_amp:
            opt.zero_grad()
            self.scaler.scale(loss).backward()
            if clip_params:
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            self.scaler.step(opt)
            self.scaler.update()
        else:
            opt.zero_grad()
            loss.backward()
            if clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()

    def _save_ckpt(self, tag: str) -> str:
        path = os.path.join(CHECKPOINT_DIR, f"ckpt_{self.condition}_{tag}.pth")
        torch.save(
            dict(
                cnn      = self.cnn.state_dict(),
                fc       = self.fc.state_dict(),
                ip       = self.ip.state_dict() if self.ip else None,
                vmf_head = self.vmf_head.state_dict(),
                condition= self.condition,
                severity = self.severity,
            ),
            path,
        )
        print(f"  ✓ checkpoint → {path}")
        return path

    def load_ckpt(self, path: str) -> None:
        obj = torch.load(path, map_location=self.device)
        self.cnn.load_state_dict(obj["cnn"])
        self.fc.load_state_dict(obj["fc"])
        if obj.get("ip"):
            self.ip = IsometricPropagator().to(self.device)
            self.ip.load_state_dict(obj["ip"])
        if obj.get("vmf_head"):
            self.vmf_head.load_state_dict(obj["vmf_head"])
        print(f"  ✓ checkpoint ← {path}")
        self,
        loader: DataLoader,
        n:      int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect up to ``n`` CNN feature vectors and their gaze labels."""
        self.cnn.eval()
        fs, gs = [], []
        with torch.no_grad():
            for imgs, gz in loader:
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        f = self.cnn(
                            imgs.to(self.device, non_blocking=True)
                        ).cpu().float().numpy()
                else:
                    f = self.cnn(
                        imgs.to(self.device, non_blocking=True)
                    ).cpu().numpy()
                fs.append(f)
                gs.append(gz.numpy())
                if sum(len(x) for x in fs) >= n:
                    break
        f = np.concatenate(fs)[:n]
        g = np.concatenate(gs)[:n]
        print(f"  Collected {len(f)} features.")
        return f, g

    def _val_fc(self, loader: DataLoader) -> float:
        self.cnn.eval()
        self.fc.eval()
        total = 0.0
        crit  = nn.L1Loss()
        with torch.no_grad():
            for imgs, gz in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                gz   = gz.to(self.device,   non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        out = self.fc(self.cnn(imgs))
                else:
                    out = self.fc(self.cnn(imgs))
                total += crit(out, gz).item()
        return total / max(len(loader), 1)

    def _vmf_val_metrics(self, val_loader_p: DataLoader) -> dict:
        self.cnn.eval()
        self.vmf_head.eval()
        nlls, angs, kappas = [], [], []

        with torch.no_grad():
            for imgs, gz_c, gz_n in val_loader_p:
                imgs = imgs.to(self.device, non_blocking=True)
                gz_n = gz_n.to(self.device, non_blocking=True)
                gz_c = gz_c.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        feats     = self.cnn(imgs)
                        mu, kappa = self.vmf_head(feats)
                else:
                    feats     = self.cnn(imgs)
                    mu, kappa = self.vmf_head(feats)

                nlls.append(self.vmf_head.nll_loss(mu, kappa, gz_n).item())
                angs.append(
                    self.vmf_head.angular_error_from_mu(mu, gz_c).cpu().numpy()
                )
                kappas.append(kappa.cpu().float().numpy().flatten())

        kappas = np.concatenate(kappas)
        angs   = np.concatenate(angs)
        return dict(
            val_nll     = float(np.mean(nlls)),
            val_ang_err = float(angs.mean()),
            kappa_mean  = float(kappas.mean()),
            kappa_min   = float(kappas.min()),
            kappa_max   = float(kappas.max()),
            cone_95_deg = cone_95(float(kappas.mean())),
        )
      
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs: int = PRETRAIN_EP,
        lr:     float = LR,
    ) -> None:

        print(f"\n{'=' * 55}\nStep 1: Pretrain ({epochs} ep)\n{'=' * 55}")
        params = list(self.cnn.parameters()) + list(self.fc.parameters())
        opt    = torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
        sch    = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=2, factor=0.5)
        crit   = nn.L1Loss()
        full_ds = train_loader.dataset

        for ep in range(epochs):
            # Sub-sample to N_PRETRAIN_SAMPLES per epoch to keep wall time manageable
            if N_PRETRAIN_SAMPLES and N_PRETRAIN_SAMPLES < len(full_ds):
                idx       = np.random.choice(
                    len(full_ds), N_PRETRAIN_SAMPLES, replace=False)
                ep_loader = self._make_loader(Subset(full_ds, idx), shuffle=True)
            else:
                ep_loader = train_loader

            self.cnn.train()
            self.fc.train()
            tloss = 0.0
            t0    = time.time()

            for i, (imgs, gz) in enumerate(ep_loader):
                imgs = imgs.to(self.device, non_blocking=True)
                gz   = gz.to(self.device,   non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = crit(self.fc(self.cnn(imgs)), gz)
                else:
                    loss = crit(self.fc(self.cnn(imgs)), gz)
                self._step(loss, opt, list(self.cnn.parameters()))
                tloss += loss.item()
                if i % 50 == 0:
                    eta = (time.time() - t0) / (i + 1) * (len(ep_loader) - i - 1)
                    print(
                        f"  ep{ep + 1} {i}/{len(ep_loader)} "
                        f"loss={loss.item():.4f}  ETA:{eta / 60:.1f}min"
                    )

            vl = self._val_fc(val_loader)
            sch.step(vl)
            print(
                f"Epoch {ep + 1}/{epochs} | "
                f"train={tloss / len(ep_loader):.4f} | "
                f"val={vl:.4f} | "
                f"{(time.time() - t0) / 60:.1f}min"
            )
            if vl < self._best_pretrain:
                self._best_pretrain = vl
                torch.save(self.cnn.state_dict(), "best_cnn.pth")
                torch.save(self.fc.state_dict(),  "best_fc.pth")

        for fname, model in [("best_cnn.pth", self.cnn), ("best_fc.pth", self.fc)]:
            if os.path.exists(fname):
                model.load_state_dict(
                    torch.load(fname, map_location=self.device))
        print(f"Pretrain done. Best val={self._best_pretrain:.4f}")
        self._save_ckpt("pretrain")
        gpu_mem()

    def build_gpm(
        self,
        train_loader: DataLoader,
        n_samples:    int = N_SAMPLES,
        n_neighbors:  int = N_NEIGHBORS,
    ) -> None:
        """Collect CNN features and fit Isomap + Sphere Alignment."""
        print(
            f"\n{'=' * 55}\n"
            f"Step 2: Build GPM (n={n_samples}, k={n_neighbors})\n"
            f"{'=' * 55}"
        )
        feats, gazs = self._collect_features(train_loader, n_samples)
        k = min(n_neighbors, len(feats) - 1)
        self.gpm = RobustGPM(
            n_neighbors=k,
            sa_epochs=SA_EPOCHS,
            n_restarts=SA_RESTARTS,
        )
        self.gpm.fit_isomap(feats)
        self.gpm.fit_sphere_alignment(self.gpm.source_pgf, gazs)
        self._save_ckpt("gpm")

    def train_ip(
        self,
        n_samples:  int   = N_SAMPLES,
        ip_epochs:  int   = IP_EP,
        lr:         float = LR,
    ) -> None:
       
        print(
            f"\n{'=' * 55}\n"
            f"Step 3: Train IP ({ip_epochs} ep, 2-phase LR)\n"
            f"{'=' * 55}"
        )
        ft = torch.FloatTensor(
            self.gpm.source_features[:n_samples]).to(self.device)
        pt = torch.FloatTensor(
            self.gpm.source_pgf[:n_samples]).to(self.device)

        self.ip = IsometricPropagator().to(self.device)
        crit    = nn.L1Loss()
        half    = ip_epochs // 2
        t0      = time.time()

        for phase, (phase_lr, phase_ep, eta_min) in enumerate(
            [(lr,     half,             1e-5),
             (lr / 2, ip_epochs - half, 1e-6)],
            start=1,
        ):
            opt = torch.optim.Adam(self.ip.parameters(), lr=phase_lr)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=phase_ep, eta_min=eta_min)

            for ep in range(phase_ep):
                self.ip.train()
                opt.zero_grad()
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = crit(self.ip(ft), pt)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    loss = crit(self.ip(ft), pt)
                    loss.backward()
                    opt.step()
                sch.step()

                global_ep = (phase - 1) * half + ep + 1
                if global_ep % max(1, ip_epochs // 10) == 0:
                    print(
                        f"  IP ep{global_ep}/{ip_epochs} | "
                        f"L1={loss.item():.6f}  (phase {phase}, lr={phase_lr})"
                    )

        self.ip.eval()
        ip_err = float(crit(self.ip(ft), pt).item())
        print(f"  IP done in {(time.time() - t0) / 60:.1f}min | "
              f"final L1={ip_err:.6f}")
        if ip_err > 0.05:
            print("  ⚠ IP L1 > 0.05 — SOT may plateau. "
                  "Check feature quality.")

    def sphere_oriented_training(
        self,
        train_loader: DataLoader,
        sot_epochs:   int   = SOT_EP,
        lr:           float = 1e-5,
    ) -> None:
        
        print(f"\n{'=' * 55}\nStep 4: SOT ({sot_epochs} ep)\n{'=' * 55}")
        for p in self.ip.parameters():
            p.requires_grad = False
        self.ip.eval()

        opt     = torch.optim.Adam(self.cnn.parameters(), lr=lr)
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=sot_epochs, eta_min=1e-6)   # FIX-2
        crit    = nn.L1Loss()
        full_ds = train_loader.dataset

        for ep in range(sot_epochs):
            if N_PRETRAIN_SAMPLES and N_PRETRAIN_SAMPLES < len(full_ds):
                idx       = np.random.choice(
                    len(full_ds), N_PRETRAIN_SAMPLES, replace=False)
                ep_loader = self._make_loader(Subset(full_ds, idx), shuffle=True)
            else:
                ep_loader = train_loader

            self.cnn.train()
            tloss = 0.0
            t0    = time.time()
            skipped = 0

            for i, (imgs, gazs) in enumerate(ep_loader):
                imgs  = imgs.to(self.device, non_blocking=True)
                ideal = self.gpm.inverse_predict(gazs).to(
                    self.device, non_blocking=True)
                if torch.isnan(ideal).any():
                    skipped += 1
                    continue
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = crit(self.ip(self.cnn(imgs)), ideal)
                else:
                    loss = crit(self.ip(self.cnn(imgs)), ideal)
                self._step(loss, opt, list(self.cnn.parameters()))
                tloss += loss.item()
                if i % 50 == 0:
                    eta_t = (time.time() - t0) / (i + 1) * (len(ep_loader) - i - 1)
                    print(
                        f"  SOT ep{ep + 1} {i}/{len(ep_loader)} "
                        f"L1={loss.item():.6f}  ETA:{eta_t / 60:.1f}min"
                    )

            sch.step()
            valid = len(ep_loader) - skipped
            suffix = f" ({skipped} NaN skipped)" if skipped else ""
            print(
                f"  SOT {ep + 1}/{sot_epochs} | "
                f"avg={tloss / max(valid, 1):.6f} | "
                f"lr={sch.get_last_lr()[0]:.2e} | "
                f"{(time.time() - t0) / 60:.1f}min{suffix}"
            )

        for p in self.ip.parameters():
            p.requires_grad = True
        print("  SOT done.")
        self._save_ckpt("sot")
        gpu_mem()

    def finetune_pathological(
        self,
        train_ds: MPIIFaceGazeDataset,
        val_ds:   MPIIFaceGazeDataset,
    ) -> tuple[DataLoader, DataLoader]:
        print(
            f"\n{'=' * 55}\n"
            f"Phase 2: Pathological fine-tuning "
            f"({self.condition}, sev={self.severity})\n"
            f"{'=' * 55}"
        )
        train_patho = NeuropalsyDataset(train_ds, self.condition, self.severity)
        val_patho   = NeuropalsyDataset(val_ds,   self.condition, self.severity)
        tl_p = self._make_loader(
            train_patho, shuffle=True,  collate_fn=neuropalsy_collate)
        vl_p = self._make_loader(
            val_patho,   shuffle=False, collate_fn=neuropalsy_collate)

        # Baseline: FC head angular error before any Phase-2 fine-tuning
        ang_before = self._fc_ang_err_on_patho(vl_p)
        print(f"\n  ┌─ GEOMETRIC ACCURACY TRACKER ──────────────────────────┐")
        print(f"  │ Stage               │ Val Ang Err (vs clean GT)       │")
        print(f"  │─────────────────────│─────────────────────────────────│")
        print(f"  │ Before FT (FC head) │ {ang_before:6.2f}°"
              f"                         │")

        print(f"\n  Stage i: vMF head only ({VMF_EPOCHS} epochs, CNN frozen)")
        for p in self.cnn.parameters():
            p.requires_grad = False

        opt_v = torch.optim.Adam(self.vmf_head.parameters(), lr=1e-4)
        sch_v = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_v, T_max=VMF_EPOCHS, eta_min=1e-5)
        logger_i   = MetricLogger("vMF-frozen")
        best_ang_i = float("inf")

        for ep in range(VMF_EPOCHS):
            self.vmf_head.train()
            tloss = 0.0
            for imgs, _, gz_n in tl_p:
                imgs = imgs.to(self.device, non_blocking=True)
                gz_n = gz_n.to(self.device, non_blocking=True)
                with torch.no_grad():
                    feats = self.cnn(imgs)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        mu, k = self.vmf_head(feats)
                        loss  = self.vmf_head.nll_loss(mu, k, gz_n)
                else:
                    mu, k = self.vmf_head(feats)
                    loss  = self.vmf_head.nll_loss(mu, k, gz_n)
                opt_v.zero_grad()
                loss.backward()
                opt_v.step()
                tloss += loss.item()
            sch_v.step()
            vm = self._vmf_val_metrics(vl_p)
            logger_i.log(ep + 1, train_nll=tloss / len(tl_p), **vm)
            if vm["val_ang_err"] < best_ang_i:
                best_ang_i = vm["val_ang_err"]
                torch.save(self.vmf_head.state_dict(), "best_vmf_i.pth")

        if os.path.exists("best_vmf_i.pth"):
            self.vmf_head.load_state_dict(
                torch.load("best_vmf_i.pth", map_location=self.device))
            print(f"  → Restored best stage-i vMF "
                  f"(ang_err={best_ang_i:.2f}°)")

        logger_i.summary_table()
        print(f"  │ After vMF stage-i   │ {best_ang_i:6.2f}°"
              f"                         │")

        print("\n  [FIX-3] Building pathological GPM on stable stage-i features ...")
        self._build_patho_gpm(tl_p, label="stage-i")
        self._save_ckpt("patho_gpm_before_joint")

        print(f"\n  Stage ii: Joint CNN + vMF ({JOINT_EPOCHS} epochs)")
        for p in self.cnn.parameters():
            p.requires_grad = True

        opt_j = torch.optim.Adam(
            list(self.cnn.parameters()) + list(self.vmf_head.parameters()),
            lr=5e-6, weight_decay=WEIGHT_DECAY,
        )
        sch_j = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_j, T_max=JOINT_EPOCHS, eta_min=1e-7)
        logger_ii   = MetricLogger("joint-CNN-vMF")
        best_ang_ii = float("inf")

        for ep in range(JOINT_EPOCHS):
            self.cnn.train()
            self.vmf_head.train()
            tloss = 0.0
            t0    = time.time()

            for i, (imgs, _, gz_n) in enumerate(tl_p):
                imgs = imgs.to(self.device, non_blocking=True)
                gz_n = gz_n.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        mu, k = self.vmf_head(self.cnn(imgs))
                        loss  = self.vmf_head.nll_loss(mu, k, gz_n)
                else:
                    mu, k = self.vmf_head(self.cnn(imgs))
                    loss  = self.vmf_head.nll_loss(mu, k, gz_n)
                self._step(loss, opt_j, list(self.cnn.parameters()))
                tloss += loss.item()
                if i % 30 == 0:
                    print(
                        f"    joint ep{ep + 1} {i}/{len(tl_p)} "
                        f"NLL={loss.item():.4f}"
                    )
            sch_j.step()
            vm = self._vmf_val_metrics(vl_p)
            logger_ii.log(
                ep + 1,
                train_nll   = tloss / len(tl_p),
                elapsed_min = (time.time() - t0) / 60,
                **vm,
            )
            if vm["val_ang_err"] < best_ang_ii:
                best_ang_ii = vm["val_ang_err"]
                self._save_ckpt("joint_best")
        best_jpath = os.path.join(
            CHECKPOINT_DIR, f"ckpt_{self.condition}_joint_best.pth")
        if os.path.exists(best_jpath):
            self.load_ckpt(best_jpath)
            print(f"  → Restored best joint ckpt "
                  f"(ang_err={best_ang_ii:.2f}°)")

        logger_ii.summary_table()
        best_row = logger_ii.best("val_ang_err")
        k_mean   = best_row.get("kappa_mean", 0.0)
        cone_deg = best_row.get("cone_95_deg", 0.0)
        print(
            f"  │ After Joint FT      │ {best_ang_ii:6.2f}°  "
            f"κ={k_mean:.2f}  95°cone={cone_deg:.1f}°       │"
        )
        print(f"  └──────────────────────────────────────────────────────────┘")

        delta = ang_before - best_ang_ii
        tag   = (
            "✓ IMPROVED"   if delta > 0.5
            else ("~ MARGINAL" if delta > 0 else "✗ NO IMPROVEMENT")
        )
        print(f"\n  📊 Geometric improvement: {delta:+.2f}°  {tag}")

        if   k_mean > KAPPA_MAX * 0.8:
            print(f"  ⚠ κ={k_mean:.1f} near ceiling — model is overconfident")
        elif k_mean < 2.0:
            print(f"  ⚠ κ={k_mean:.1f} very low — high uncertainty "
                  "(expected for severe pathology)")
        else:
            print(f"  ✓ κ={k_mean:.1f} in healthy range")

        self._save_ckpt("phase2_final")
        return tl_p, vl_p
      
    def _fc_ang_err_on_patho(self, val_loader_p: DataLoader) -> float:
        """Baseline angular error of the FC head vs clean GT labels."""
        self.cnn.eval()
        self.fc.eval()
        errs = []
        with torch.no_grad():
            for imgs, gz_c, _ in val_loader_p:
                imgs = imgs.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        pred = self.fc(self.cnn(imgs)).cpu().float().numpy()
                else:
                    pred = self.fc(self.cnn(imgs)).cpu().numpy()
                pred /= np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8
                errs.append(angular_error_np(pred, gz_c.numpy()))
        return float(np.concatenate(errs).mean())

    def _build_patho_gpm(
        self,
        train_loader_p: DataLoader,
        label: str = "",
    ) -> None:
  
        self.cnn.eval()
        fp, gn = [], []
        with torch.no_grad():
            for imgs, _, gz_n in train_loader_p:
                imgs = imgs.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        f = self.cnn(imgs).cpu().float().numpy()
                else:
                    f = self.cnn(imgs).cpu().numpy()
                fp.append(f)
                gn.append(gz_n.numpy())
                if sum(len(x) for x in fp) >= N_SAMPLES:
                    break

        fp = np.concatenate(fp)[:N_SAMPLES]
        gn = np.concatenate(gn)[:N_SAMPLES]
        k  = min(N_NEIGHBORS, len(fp) - 1)

        self.gpm_patho = RobustGPM(
            n_neighbors=k,
            sa_epochs=SA_EPOCHS,
            n_restarts=SA_RESTARTS,
        )
        self.gpm_patho.fit_isomap(fp)
        self.gpm_patho.fit_sphere_alignment(self.gpm_patho.source_pgf, gn)

        sa_err = angular_error_np(
            self.gpm_patho.predict(self.gpm_patho.source_pgf), gn
        ).mean()
        flag = "✓" if sa_err < 10 else ("⚠" if sa_err < 14 else "✗")
        print(f"  Patho GPM SA in-sample ({label}): {sa_err:.2f}° {flag}")

    def evaluate_dual(
        self,
        val_loader_clean: DataLoader,
        val_loader_patho: DataLoader,
    ) -> dict:

        print(f"\n{'=' * 55}\nDual Evaluation\n{'=' * 55}")
        self.cnn.eval()
        results: dict = {}
        print("\n  [A] Healthy model ...")
        fc_l, gz_l, ft_l = [], [], []
        with torch.no_grad():
            for imgs, gz in val_loader_clean:
                imgs = imgs.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        f   = self.cnn(imgs).cpu().float().numpy()
                        pf  = self.fc(
                            torch.FloatTensor(f).to(self.device)
                        ).cpu().numpy()
                else:
                    f   = self.cnn(imgs).cpu().numpy()
                    pf  = self.fc(
                        torch.FloatTensor(f).to(self.device)
                    ).cpu().numpy()
                pf /= np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8
                ft_l.append(f)
                gz_l.append(gz.numpy())
                fc_l.append(pf)

        feats_c = np.concatenate(ft_l)
        gazes_c = np.concatenate(gz_l)
        fc_pred = np.concatenate(fc_l)
        fc_err  = angular_error_np(fc_pred, gazes_c)
        results.update(fc=fc_err, fc_pred=fc_pred, gt_clean=gazes_c)
        print(
            f"  FC  Layer: {fc_err.mean():.2f}° ± {fc_err.std():.2f}°  "
            f"med={np.median(fc_err):.2f}°"
        )

        if self.gpm:
            pgf_c = self.gpm.project_all_to_3d(feats_c)
            gp    = self.gpm.predict(pgf_c)
            ge    = angular_error_np(gp, gazes_c)
            impr  = (fc_err.mean() - ge.mean()) / fc_err.mean() * 100
            results.update(gpm_healthy=ge, gpm_healthy_pred=gp)
            print(
                f"  GPM healthy: {ge.mean():.2f}° ± {ge.std():.2f}°  "
                f"({impr:+.1f}%)"
            )

        print("\n  [B] Pathological model ...")
        fp_l, gc_l, gn_l, mu_l, ka_l = [], [], [], [], []
        with torch.no_grad():
            for imgs, gz_c, gz_n in val_loader_patho:
                imgs = imgs.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        f         = self.cnn(imgs).cpu().float().numpy()
                        mu, kappa = self.vmf_head(
                            torch.FloatTensor(f).to(self.device))
                else:
                    f         = self.cnn(imgs).cpu().numpy()
                    mu, kappa = self.vmf_head(
                        torch.FloatTensor(f).to(self.device))
                fp_l.append(f)
                gc_l.append(gz_c.numpy())
                gn_l.append(gz_n.numpy())
                mu_l.append(mu.cpu().numpy())
                ka_l.append(kappa.cpu().float().numpy().flatten())

        feats_p   = np.concatenate(fp_l)
        gazes_p   = np.concatenate(gc_l)
        gazes_n   = np.concatenate(gn_l)
        mu_all    = np.concatenate(mu_l)
        kappa_all = np.concatenate(ka_l)

        vmf_ec = angular_error_np(mu_all, gazes_p)   # vs clean GT (key metric)
        vmf_en = angular_error_np(mu_all, gazes_n)   # vs noisy GT
        mk     = float(kappa_all.mean())

        results.update(
            vmf_pred=mu_all, vmf_kappa=kappa_all,
            vmf_err_vs_clean=vmf_ec, vmf_err_vs_noisy=vmf_en,
            gt_noisy=gazes_n,
        )
        print(
            f"  vMF μ (vs clean): {vmf_ec.mean():.2f}° ± {vmf_ec.std():.2f}°  "
            f"med={np.median(vmf_ec):.2f}°\n"
            f"  vMF μ (vs noisy): {vmf_en.mean():.2f}° ± {vmf_en.std():.2f}°\n"
            f"  κ mean/min/max  : {mk:.2f} / "
            f"{float(kappa_all.min()):.2f} / {float(kappa_all.max()):.2f}\n"
            f"  95%% cone       : {cone_95(mk):.1f}°"
        )

        if self.gpm_patho:
            pgf_p = self.gpm_patho.project_all_to_3d(feats_p)
            gp_p  = self.gpm_patho.predict(pgf_p)
            ge_p  = angular_error_np(gp_p, gazes_n)
            results.update(gpm_patho=ge_p, gpm_patho_pred=gp_p)
            print(f"  GPM patho: {ge_p.mean():.2f}° ± {ge_p.std():.2f}°")

        print(f"\n  {'─' * 58}")
        print(f"  {'Method':<28} {'Mean':>8} {'Std':>8} {'Median':>8}")
        print(f"  {'─' * 58}")
        for name, arr in [
            ("FC baseline (healthy)",   fc_err),
            ("GPM-AGG (healthy)",        results.get("gpm_healthy", np.zeros(1))),
            ("GPM-AGG (pathological)",   results.get("gpm_patho",   np.zeros(1))),
            ("vMF μ vs clean GT",        vmf_ec),
            ("vMF μ vs noisy GT",        vmf_en),
        ]:
            print(
                f"  {name:<28} {arr.mean():>7.2f}° "
                f"{arr.std():>7.2f}° {np.median(arr):>7.2f}°"
            )
        print(f"  {'─' * 58}")
        print(
            f"  κ: {mk:.2f}  "
            f"(min={float(kappa_all.min()):.2f}  "
            f"max={float(kappa_all.max()):.2f})  "
            f"95°cone={cone_95(mk):.1f}°"
        )
        print(f"  {'─' * 58}")
        return results

    def export_results(
        self,
        results: dict,
        path:    str = "",
        n_vis:   int = 200,
    ) -> dict:
        if not path:
            path = RESULTS_JSON_TPL.format(condition=self.condition)

        out = dict(
            condition   = self.condition,
            severity    = self.severity,
            kappa_clamp = [KAPPA_MIN, KAPPA_MAX],
            sa_restarts = SA_RESTARTS,
            sa_epochs   = SA_EPOCHS,
            summary     = {},
            samples_healthy      = [],
            samples_pathological = [],
        )
        s = out["summary"]

        if "fc" in results:
            fe = results["fc"]
            s.update(
                fc_mean   = round(float(fe.mean()),       3),
                fc_std    = round(float(fe.std()),        3),
                fc_median = round(float(np.median(fe)),   3),
            )
        if "gpm_healthy" in results:
            ge = results["gpm_healthy"]
            s.update(
                gpm_healthy_mean = round(float(ge.mean()), 3),
                gpm_healthy_std  = round(float(ge.std()),  3),
                improvement_pct  = round(float(
                    (results["fc"].mean() - ge.mean())
                    / results["fc"].mean() * 100), 2),
            )
        if "gpm_patho" in results:
            gp = results["gpm_patho"]
            s.update(
                gpm_patho_mean = round(float(gp.mean()), 3),
                gpm_patho_std  = round(float(gp.std()),  3),
            )
        if "vmf_kappa" in results:
            ka = results["vmf_kappa"]
            s.update(
                vmf_kappa_mean    = round(float(ka.mean()),                    3),
                vmf_kappa_min     = round(float(ka.min()),                     3),
                vmf_kappa_max     = round(float(ka.max()),                     3),
                vmf_cone_95_deg   = round(cone_95(float(ka.mean())),           2),
                vmf_err_vs_clean  = round(float(
                    results["vmf_err_vs_clean"].mean()),  3),
                vmf_err_vs_noisy  = round(float(
                    results["vmf_err_vs_noisy"].mean()),  3),
            )

        if all(k in results for k in
               ("gt_clean", "gpm_healthy_pred", "fc_pred",
                "gpm_healthy", "fc")):
            n = min(n_vis, len(results["gt_clean"]))
            out["samples_healthy"] = [
                dict(
                    id      = int(i),
                    gt      = [round(float(v), 4) for v in results["gt_clean"][i]],
                    gpm     = [round(float(v), 4) for v in results["gpm_healthy_pred"][i]],
                    fc      = [round(float(v), 4) for v in results["fc_pred"][i]],
                    gpm_err = round(float(results["gpm_healthy"][i]), 2),
                    fc_err  = round(float(results["fc"][i]),          2),
                )
                for i in range(n)
            ]

        if all(k in results for k in
               ("gt_noisy", "vmf_pred", "vmf_kappa",
                "vmf_err_vs_noisy", "vmf_err_vs_clean")):
            n  = min(n_vis, len(results["gt_noisy"]))
            ka = results["vmf_kappa"]
            out["samples_pathological"] = [
                dict(
                    id           = int(i),
                    gt_noisy     = [round(float(v), 4) for v in results["gt_noisy"][i]],
                    vmf_mu       = [round(float(v), 4) for v in results["vmf_pred"][i]],
                    kappa        = round(float(ka[i]),            3),
                    cone_95_deg  = round(cone_95(float(ka[i])),   2),
                    err_vs_noisy = round(float(results["vmf_err_vs_noisy"][i]), 2),
                    err_vs_clean = round(float(results["vmf_err_vs_clean"][i]), 2),
                )
                for i in range(n)
            ]

        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  Exported → {path}")
        return out

    def run_full_pipeline(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        train_ds:     MPIIFaceGazeDataset,
        val_ds:       MPIIFaceGazeDataset,
    ) -> dict:
        
        print(
            "\n" + "=" * 55 +
            f"\nDUAL AGG | {self.condition} | {DEVICE}\n" +
            "=" * 55
        )
        t0    = time.time()
        eff_n = min(N_SAMPLES, len(train_ds))

        self.pretrain(train_loader, val_loader)
        self.build_gpm(train_loader, n_samples=eff_n, n_neighbors=N_NEIGHBORS)
        self.train_ip(n_samples=eff_n)
        self.sphere_oriented_training(train_loader)
        tl_p, vl_p = self.finetune_pathological(train_ds, val_ds)
        results     = self.evaluate_dual(val_loader, vl_p)

        print(f"\n✓ Done: {(time.time() - t0) / 60:.1f} min total")
        self.export_results(results)
        self._save_ckpt("final")
        return results
