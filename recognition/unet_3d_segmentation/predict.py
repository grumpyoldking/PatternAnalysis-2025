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
    affine = np.diag([Dx, Dy, Dz, 1.0])  # RAS-ish spacing on diag
    data_xyz = np.transpose(mask_dhw, (2, 1, 0)).astype(np.int16, copy=False)  # (D,H,W)->(X,Y,Z)
    nib.save(nib.Nifti1Image(data_xyz, affine), str(out_path))

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
        imgs   = batch["image"].to(device, non_blocking=True)          # [B,1,D,H,W]
        labels = batch["label"][:,0].long().to(device, non_blocking=True)  # [B,D,H,W]
        with amp.autocast('cuda', dtype=amp_dtype):
            logits = model(imgs)                                        # [B,C,D,H,W]
        pred = torch.argmax(logits, dim=1)                               # [B,D,H,W]

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
    ap = argparse.ArgumentParser(description="3D U-Net inference")
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
    _, val_loader = make_loaders_for_hipmri(
        root=root,
        target_spacing=(2.0, 2.0, 2.0),
        patch_size=None,              # full volume
        batch_size=1,
        workers=args.workers
    )

    # Model (match training config)
    model = UNet3D(
        in_channels=1,
        out_channels=args.num_classes,
        features=(32, 64, 128, 256, 512),
        norm='instance',
        dropout=0.1
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
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
                logits = model(imgs)                                 # [1,C,D,H,W]
                pred = torch.argmax(logits, dim=1).squeeze(0)        # [D,H,W]
            pred_np = pred.cpu().numpy()

            # Save NIfTI
            out_nii = outdir / "nii" / f"{cid}_pred.nii.gz"
            save_nii_mask(pred_np, spacing, out_nii)
            print(f"Saved: {out_nii}")

            # Optional overlays (first N cases)
            if count < args.viz:
                vol = batch["image"][0, 0].cpu().numpy()
                z = best_slice(pred_np)
                sl = window_img(vol[z])
                lbl = pred_np[z]
                plt.figure(figsize=(8, 3))
                ax1 = plt.subplot(1, 2, 1); ax1.imshow(sl, cmap="gray"); ax1.axis("off"); ax1.set_title(f"{cid} | z={z}")
                ax2 = plt.subplot(1, 2, 2); ax2.imshow(sl, cmap="gray")
                ax2.imshow(np.ma.masked_where(lbl == 0, lbl), alpha=0.45, cmap="tab20"); ax2.axis("off"); ax2.set_title("Prediction")
                plt.tight_layout()
                png_path = outdir / "viz" / f"{cid}_z{z:03d}.png"
                plt.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close()
                print(f"Saved: {png_path}")
            count += 1

    # ---------------- Dice requirement check ----------------
    report = dice_report(
        model=model,
        loader=val_loader,  # swap to your real test loader when you have one
        device=device,
        num_classes=args.num_classes,
        amp_dtype=amp_dtype,
        label_names={0: "background", 1: "PZ", 2: "TZ", 3: "SV", 4: "bladder", 5: "rectum"},
        threshold=args.threshold
    )

    print("Done. Pass requirement:", report["pass"])

if __name__ == "__main__":
    main()
