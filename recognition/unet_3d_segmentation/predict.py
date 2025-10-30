# predict.py
import argparse
from pathlib import Path
import os
import re
import sys
import numpy as np
import torch
from torch import amp
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn as nn

# -------------------- helpers --------------------

def find_dataset_root() -> str:
    # Mirror train.py behavior, but allow override via env
    env_root = os.getenv("DATASET_ROOT")
    if env_root and (Path(env_root) / "semantic_MRs_anon").is_dir():
        return env_root

    gd_specific = Path("/content/drive/MyDrive/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data")
    if (gd_specific / "semantic_MRs_anon").is_dir():
        return str(gd_specific)

    # fallback: current folder tree
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for cand in [here, *here.rglob("*")]:
        if not cand.is_dir():
            continue
        if (cand / "semantic_MRs_anon").is_dir() and (cand / "semantic_labels_anon").is_dir():
            return str(cand)
    raise FileNotFoundError("Dataset root not found. Set DATASET_ROOT or mount Google Drive.")

def case_root(case_id: str) -> str:
    m = re.match(r"^(Case_\d+)", case_id)
    return m.group(1) if m else case_id.split("_")[0]

def window_img(x):
    p2, p98 = np.percentile(x, (2, 98))
    return np.clip((x - p2) / (p98 - p2 + 1e-6), 0, 1)

def best_slice(lbl_3d: np.ndarray) -> int:
    area = (lbl_3d > 0).reshape(lbl_3d.shape[0], -1).sum(axis=1)
    return int(area.argmax()) if area.max() > 0 else lbl_3d.shape[0] // 2

def save_nii_mask(mask_dhw: np.ndarray, spacing_dhw, out_path: Path):
    """Save integer prediction (D,H,W) as NIfTI with diagonal affine using spacing."""
    Dz, Dy, Dx = [float(s) for s in spacing_dhw]
    affine = np.diag([Dx, Dy, Dz, 1.0])  # simple spacing-only affine
    data_xyz = np.transpose(mask_dhw, (2, 1, 0)).astype(np.int16, copy=False)  # (D,H,W)->(X,Y,Z)
    nib.save(nib.Nifti1Image(data_xyz, affine), str(out_path))

# -------------------- Improved UNet (3D) --------------------
# Residual blocks + SCSE attention + attention-gated skips + learnable down/upsampling

class ConvNormAct3d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8, act="silu"):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "lrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, dropout=0.0, groups=8, act="silu"):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv1 = ConvNormAct3d(in_ch, mid_ch, k=3, s=1, p=1, groups=groups, act=act)
        self.drop = nn.Dropout3d(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.conv2 = ConvNormAct3d(mid_ch, out_ch, k=3, s=1, p=1, groups=groups, act=act)
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return y + self.proj(x)

class SCSE3d(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        red = max(ch // r, 1)
        self.cse_avg = nn.AdaptiveAvgPool3d(1)
        self.cse_fc = nn.Sequential(
            nn.Conv3d(ch, red, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(red, ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.sse = nn.Sequential(nn.Conv3d(ch, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        c = self.cse_fc(self.cse_avg(x))
        s = self.sse(x)
        return x * c + x * s

class AttGate3d(nn.Module):
    def __init__(self, in_skip, in_g, inter):
        super().__init__()
        self.theta = nn.Conv3d(in_skip, inter, 1, bias=False)
        self.phi   = nn.Conv3d(in_g,    inter, 1, bias=False)
        self.act   = nn.ReLU(inplace=True)
        self.psi   = nn.Sequential(nn.Conv3d(inter, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, skip, g):
        if skip.shape[2:] != g.shape[2:]:
            g = nn.functional.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        a = self.act(self.theta(skip) + self.phi(g))
        att = self.psi(a)
        return skip * att

class DownBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, groups=8, act="silu"):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act  = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)
        self.block = ResBlock3d(out_ch, out_ch, dropout=dropout, groups=groups, act=act)
        self.scse  = SCSE3d(out_ch)

    def forward(self, x):
        x = self.act(self.norm(self.down(x)))
        x = self.block(x)
        x = self.scse(x)
        return x

class UpBlock3d(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0, groups=8, act="silu", up_mode="trilinear"):
        super().__init__()
        self.up_mode = up_mode
        if up_mode == "deconv":
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
            up_out = out_ch
        else:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.reduce = nn.Conv3d(in_ch, out_ch, 1, bias=False)
            up_out = out_ch
        self.gate = AttGate3d(skip_ch, up_out, inter=max(out_ch // 2, 8))
        self.fuse = ResBlock3d(out_ch + skip_ch, out_ch, dropout=dropout, groups=groups, act=act)
        self.scse = SCSE3d(out_ch)

    def forward(self, x, skip):
        if self.up_mode == "deconv":
            x = self.up(x)
        else:
            x = self.reduce(self.up(x))
        skip = self.gate(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.scse(x)
        return x

class ImprovedUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features=(32, 64, 128, 256, 512),
        dropout: float = 0.1,
        groups: int = 8,
        act: str = "silu",
        up_mode: str = "trilinear",
        deep_supervision: bool = False,
    ):
        super().__init__()
        assert len(features) >= 4, "Use at least 4 stages for 3D volumes."
        self.deep_supervision = deep_supervision
        chans = list(features)

        self.stem = ResBlock3d(in_channels, chans[0], dropout=dropout, groups=groups, act=act)
        self.stem_scse = SCSE3d(chans[0])

        self.enc = nn.ModuleList()
        for i in range(len(chans) - 1):
            self.enc.append(DownBlock3d(chans[i], chans[i+1], dropout=dropout, groups=groups, act=act))

        self.dec = nn.ModuleList()
        for i in reversed(range(len(chans) - 1)):
            in_ch  = chans[i+1]
            skip_ch= chans[i]
            out_ch = chans[i]
            self.dec.append(UpBlock3d(in_ch, skip_ch, out_ch, dropout=dropout, groups=groups, act=act, up_mode=up_mode))

        self.head = nn.Conv3d(chans[0], out_channels, 1, bias=True)

        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv3d(chans[i], out_channels, 1, bias=True) for i in range(1, len(chans))
            ])

    def forward(self, x):
        s0 = self.stem_scse(self.stem(x))
        feats = [s0]
        y = s0
        for down in self.enc:
            y = down(y)
            feats.append(y)

        aux = []
        for i, up in enumerate(self.dec):
            skip = feats[-(i+2)]
            y = up(y, skip)
            if self.deep_supervision and i < len(self.dec) - 1:
                aux.append(y)

        logits = self.head(y)
        if not self.deep_supervision:
            return logits

        aux_logits = []
        for i, y_i in enumerate(aux):
            scale = 2 ** (i + 1)
            up = nn.functional.interpolate(y_i, scale_factor=scale, mode="trilinear", align_corners=False)
            aux_logits.append(self.aux_heads[-(i+2)](up))
        return logits, aux_logits

# -------------------- Dice evaluator (mean per class over set) --------------------

@torch.no_grad()
def dice_report(model, loader, device, num_classes: int, amp_dtype, label_names=None, threshold=0.7):
    """
    Computes mean Dice per class across the dataset.
    - Skips items where a class is absent in GT (denominator==0).
    - Prints a summary and returns a dict.
    """
    model.eval()
    sums = torch.zeros(num_classes, device=device)
    cnts = torch.zeros(num_classes, device=device)

    for batch in loader:
        imgs   = batch["image"].to(device, non_blocking=True)              # [B,1,D,H,W]
        labels = batch["label"][:,0].long().to(device, non_blocking=True)  # [B,D,H,W]
        with amp.autocast('cuda', dtype=amp_dtype):
            out = model(imgs)
            logits = out[0] if isinstance(out, (tuple, list)) else out     # handle deep supervision
        pred = torch.argmax(logits, dim=1)                                  # [B,D,H,W]

        for c in range(num_classes):
            p = (pred == c).float()
            t = (labels == c).float()
            inter = (p * t).flatten(1).sum(1)
            den   = p.flatten(1).sum(1) + t.flatten(1).sum(1)
            valid = den > 0
            if valid.any():
                dice = (2 * inter[valid]) / (den[valid] + 1e-6)
                sums[c] += dice.mean()
                cnts[c] += 1

    mean_dice = torch.where(cnts > 0, sums / cnts.clamp_min(1), torch.zeros_like(sums))
    per_class = mean_dice.tolist()
    per_class_no_bg = per_class[1:] if num_classes > 1 else per_class
    min_no_bg = float(min(per_class_no_bg)) if per_class_no_bg else 0.0

    # Pretty print
    print("\n=== Dice report (mean over set) ===")
    for c in range(num_classes):
        name = (label_names.get(c, f"class_{c}") if isinstance(label_names, dict) else f"class_{c}")
        status = ""
        if c != 0:
            if cnts[c] == 0:
                status = " (absent in GT)"
            else:
                status = " ✅" if mean_dice[c] >= threshold else " ❌"
        print(f"  {c:2d} [{name:>12}]: {mean_dice[c]:.4f} over {int(cnts[c].item())} imgs{status}")
    print(f"Min Dice (excluding background): {min_no_bg:.4f}  —  "
          f"{'PASS ✅' if min_no_bg >= threshold else 'FAIL ❌'} (threshold={threshold})\n")

    return {
        "per_class": per_class,
        "counts": [int(x) for x in cnts.tolist()],
        "min_excluding_bg": min_no_bg,
        "threshold": threshold,
        "pass": (min_no_bg >= threshold),
    }

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Improved 3D U-Net inference")
    ap.add_argument("--ckpt", type=str, default="runs/checkpoints/best.pt",
                    help="Path to checkpoint .pt (default: runs/checkpoints/best.pt)")
    ap.add_argument("--outdir", type=str, default="runs/preds",
                    help="Folder to write NIfTI masks and PNG overlays")
    ap.add_argument("--num-classes", type=int, default=6,
                    help="Number of output classes (incl. background)")
    ap.add_argument("--viz", type=int, default=5, help="How many overlays to save (0 to disable)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="Dice threshold to pass (excluding background)")
    # tolerate extra IPython args like "-f <kernel.json>"
    args, _unknown = ap.parse_known_args()

    # CUDA / AMP
    assert torch.cuda.is_available(), "CUDA GPU not found."
    device = torch.device("cuda")
    cc_major, _ = torch.cuda.get_device_capability(0)
    amp_dtype = torch.bfloat16 if cc_major >= 8 else torch.float16

    # Data: use validation loader (full volumes padded)
    root = find_dataset_root()
    # NOTE: make_loaders_for_hipmri must exist in your project and accept these params.
    _, val_loader = make_loaders_for_hipmri(
        root=root,
        target_spacing=(2.0, 2.0, 2.0),
        patch_size=None,              # full volume
        batch_size=1,
        workers=args.workers
    )

    # Model (match training config)
    model = ImprovedUNet3D(
        in_channels=1,
        out_channels=args.num_classes,
        features=(32, 64, 128, 256, 512),
        dropout=0.1,
        groups=8,
        act="silu",
        up_mode="trilinear",
        deep_supervision=False  # if training used True, we still consume logits[0] below
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)  # tolerate state_dict at top-level
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("load_state_dict notes:", {"missing": missing, "unexpected": unexpected})
    model.eval()

    outdir = Path(args.outdir)
    (outdir / "nii").mkdir(parents=True, exist_ok=True)
    (outdir / "viz").mkdir(parents=True, exist_ok=True)

    # ---------------- Inference pass (saves nii + optional overlays) ----------------
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device, non_blocking=True)      # [1,1,D,H,W]
            spacing = batch["spacing"][0].cpu().numpy()              # (Dz,Dy,Dx)
            cid = batch["id"][0]
            with amp.autocast('cuda', dtype=amp_dtype):
                out = model(imgs)                                    # logits or (logits, aux)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                pred = torch.argmax(logits, dim=1).squeeze(0)        # [D,H,W]
            pred_np = pred.cpu().numpy()

            # Save NIfTI
            out_nii = outdir / "nii" / f"{cid}_pred.nii.gz"
            save_nii_mask(pred_np, spacing, out_nii)
            print(f"Saved: {out_nii}")

            # Optional overlays (first N cases)
            if count < args.viz:
                vol = batch["image"][0, 0].cpu().numpy()             # [D,H,W]
                gt  = batch["label"][0, 0].cpu().numpy().astype(int) # [D,H,W]

                # choose a slice with GT content when possible
                z = best_slice(gt if gt.max() > 0 else pred_np)

                sl = window_img(vol[z])
                gt_sl = gt[z]
                pr_sl = pred_np[z]

                # --- 3-panel: image+GT, image+Pred, (for quick eyeballing) ---
                plt.figure(figsize=(12, 3.4))
                ax1 = plt.subplot(1, 3, 1); ax1.imshow(sl, cmap="gray"); ax1.axis("off"); ax1.set_title(f"{cid} | GT (z={z})")
                ax1.imshow(np.ma.masked_where(gt_sl == 0, gt_sl), alpha=0.45, cmap="tab20")

                ax2 = plt.subplot(1, 3, 2); ax2.imshow(sl, cmap="gray"); ax2.axis("off"); ax2.set_title("Prediction")
                ax2.imshow(np.ma.masked_where(pr_sl == 0, pr_sl), alpha=0.45, cmap="tab20")

                # quick slice Dice (excluding background)
                pr_flat = pr_sl.reshape(-1)
                gt_flat = gt_sl.reshape(-1)
                nonbg = gt_flat > 0
                if nonbg.any():
                    inter = np.sum((pr_flat == gt_flat) & nonbg)
                    den = np.sum(nonbg) + np.sum(pr_flat > 0)
                    slice_dice = (2.0 * inter) / (den + 1e-6)
                else:
                    slice_dice = 0.0

                # --- Contour compare: GT (green) vs Pred (red) on same image ---
                ax3 = plt.subplot(1, 3, 3); ax3.imshow(sl, cmap="gray"); ax3.axis("off")
                vals_gt = [v for v in np.unique(gt_sl) if v != 0]
                vals_pr = [v for v in np.unique(pr_sl) if v != 0]
                if vals_gt:
                    ax3.contour(gt_sl, levels=vals_gt, linewidths=1.0, colors='g')
                if vals_pr:
                    ax3.contour(pr_sl, levels=vals_pr, linewidths=1.0, colors='r')
                ax3.set_title(f"Contours (GT=green, Pred=red)\nSlice Dice≈{slice_dice:.3f}")

                plt.tight_layout()
                png_path = outdir / "viz" / f"{cid}_z{z:03d}_gt_vs_pred.png"
                plt.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close()
                print(f"Saved: {png_path}")

                # (Optional) simpler two-panel already covered above; also add single overlay compare image if you want:
                plt.figure(figsize=(6, 3.2))
                ax = plt.subplot(1, 2, 1); ax.imshow(sl, cmap="gray"); ax.axis("off"); ax.set_title("GT")
                ax.imshow(np.ma.masked_where(gt_sl == 0, gt_sl), alpha=0.45, cmap="tab20")
                ax = plt.subplot(1, 2, 2); ax.imshow(sl, cmap="gray"); ax.axis("off"); ax.set_title("Pred")
                ax.imshow(np.ma.masked_where(pr_sl == 0, pr_sl), alpha=0.45, cmap="tab20")
                plt.tight_layout()
                png2 = outdir / "viz" / f"{cid}_z{z:03d}_overlay_compare.png"
                plt.savefig(png2, dpi=150, bbox_inches="tight"); plt.close()
                print(f"Saved: {png2}")

            count += 1

    # ---------------- Dice requirement check ----------------
    report = dice_report(
        model=model,
        loader=val_loader,  # swap to your real test loader when you have one
        device=device,
        num_classes=args.num_classes,
        amp_dtype=amp_dtype,
        label_names={0: "class 0", 1: "class 1", 2: "class 2", 3: "class 3", 4: "class 4", 5: "class 5"},
        threshold=args.threshold
    )

    print("Done. Pass requirement:", report["pass"])

if __name__ == "__main__":
    main()
