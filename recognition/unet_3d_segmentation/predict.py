# predict.py
import argparse
from pathlib import Path
import os
import re
import numpy as np
import torch
from torch import amp
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn as nn

# -------------------- helpers --------------------

def find_dataset_root() -> str:
    """
    Locate dataset root that contains semantic_MRs_anon/ (and ideally semantic_labels_anon/).
    Priority:
      1) DATASET_ROOT env var
      2) Known Google Drive path (Colab)
      3) Scan the current repo tree
    """
    env_root = os.getenv("DATASET_ROOT")
    if env_root and (Path(env_root) / "semantic_MRs_anon").is_dir():
        return env_root

    gd_specific = Path("/content/drive/MyDrive/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data")
    if (gd_specific / "semantic_MRs_anon").is_dir():
        return str(gd_specific)

    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for cand in [here, *here.rglob("*")]:
        if not cand.is_dir():
            continue
        if (cand / "semantic_MRs_anon").is_dir() and (cand / "semantic_labels_anon").is_dir():
            return str(cand)

    raise FileNotFoundError("Dataset root not found. Set DATASET_ROOT or mount Google Drive.")

def case_root(case_id: str) -> str:
    """Reduce 'Case_004_Week0' → 'Case_004' for grouping / pretty printing."""
    m = re.match(r"^(Case_\d+)", case_id)
    return m.group(1) if m else case_id.split("_")[0]

def window_img(x: np.ndarray):
    """Window to [2nd,98th] percentiles and normalize to [0,1] for visualization."""
    p2, p98 = np.percentile(x, (2, 98))
    return np.clip((x - p2) / (p98 - p2 + 1e-6), 0, 1)

def best_slice(lbl_3d: np.ndarray) -> int:
    """
    Choose an axial slice index that maximizes foreground area;
    fall back to the middle slice if the mask is empty.
    """
    area = (lbl_3d > 0).reshape(lbl_3d.shape[0], -1).sum(axis=1)
    return int(area.argmax()) if area.max() > 0 else lbl_3d.shape[0] // 2

def save_nii_mask(mask_dhw: np.ndarray, spacing_dhw, out_path: Path):
    """
    Save integer (D,H,W) prediction as NIfTI, using a diagonal affine derived from spacing.
    NOTE: This assumes canonical axis order and ignores orientation (sufficient for quick export).
    """
    Dz, Dy, Dx = [float(s) for s in spacing_dhw]
    affine = np.diag([Dx, Dy, Dz, 1.0])  # spacing along X,Y,Z; homogeneous coord in last column
    data_xyz = np.transpose(mask_dhw, (2, 1, 0)).astype(np.int16, copy=False)  # (D,H,W) -> (X,Y,Z)
    nib.save(nib.Nifti1Image(data_xyz, affine), str(out_path))

# -------------------- Improved UNet (3D) --------------------
# (Assumes ImprovedUNet3D is already defined in the notebook.)

# -------------------- Dice evaluator (mean per class over set) --------------------

@torch.no_grad()
def dice_report(model, loader, device, num_classes: int, amp_dtype, label_names=None, threshold=0.7):
    """
    Compute mean Dice per class across the provided loader.
    - Skips samples where a class is absent in the GT (denominator==0).
    - Prints a PASS/FAIL using the minimum foreground Dice vs threshold.
    """
    model.eval()
    sums = torch.zeros(num_classes, device=device)  # sum of per-batch dice per class
    cnts = torch.zeros(num_classes, device=device)  # number of batches contributing to each class

    for batch in loader:
        imgs   = batch["image"].to(device, non_blocking=True)              # [B,1,D,H,W]
        labels = batch["label"][:,0].long().to(device, non_blocking=True)  # [B,D,H,W]
        with amp.autocast('cuda', dtype=amp_dtype):
            out = model(imgs)
            logits = out[0] if isinstance(out, (tuple, list)) else out     # support deep supervision
        pred = torch.argmax(logits, dim=1)                                  # [B,D,H,W]

        # Per-class Dice accumulation
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

    # Human-readable summary
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
    """
    Predict masks on the TEST split:
      - Uses ImprovedUNet3D (already defined in notebook) and loaded weights
      - Iterates over full volumes (padded) from the dataset
      - Saves NIfTI predictions and optional 2D overlays
      - Prints a dataset-level Dice report vs threshold
    """
    ap = argparse.ArgumentParser(description="Improved 3D U-Net inference (TEST split)")
    ap.add_argument("--ckpt", type=str, default="runs/checkpoints/best.pt",
                    help="Path to checkpoint .pt (default: runs/checkpoints/best.pt)")
    ap.add_argument("--outdir", type=str, default="runs/preds",
                    help="Folder to write NIfTI masks and PNG overlays")
    ap.add_argument("--num-classes", type=int, default=6,
                    help="Number of output classes (incl. background)")
    ap.add_argument("--viz", type=int, default=5, help="How many overlays to save (0 to disable)")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="Dice threshold to pass (excluding background)")
    args, _unknown = ap.parse_known_args()  # tolerate extra IPython args in Colab

    # ---- CUDA / AMP ----
    assert torch.cuda.is_available(), "CUDA GPU not found."
    device = torch.device("cuda")
    cc_major, _ = torch.cuda.get_device_capability(0)
    amp_dtype = torch.bfloat16 if cc_major >= 8 else torch.float16  # bf16 on Ampere+, else fp16

    # ---- Data: TEST split (full volumes, padded) ----
    root = find_dataset_root()

    # NOTE: Prostate3DDataset is assumed to be defined in the same notebook/session.
    test_ds = Prostate3DDataset(
        root=root,
        split="test",
        img_dirnames=("semantic_MRs_anon", "images", "imagesTr"),
        lbl_dirnames=("semantic_labels_anon", "labels", "labelsTr"),
        target_spacing=(2.0, 2.0, 2.0),
        crop_to_foreground=False,
        patch_size=None,              # full volume (dataset handles pad-to-multiple)
        augment=False,
        for_eval_full_volume=True
    )

    # Minimal collate: just stack tensors and carry metadata
    def collate_pad(batch):
        return {
            "image": torch.stack([b["image"] for b in batch], dim=0),
            "label": torch.stack([b["label"] for b in batch], dim=0),
            "id":    [b["id"] for b in batch],
            "spacing": torch.stack([b["spacing"] for b in batch], dim=0),
        }

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_pad
    )
    print(f"Discovered TEST set cases: {len(test_ds)} (volumes)")

    # ---- Model (mirror training config) ----
    # NOTE: ImprovedUNet3D is assumed to be defined in the same notebook/session.
    model = ImprovedUNet3D(
        in_channels=1,
        out_channels=args.num_classes,
        features=(32, 64, 128, 256, 512),
        dropout=0.1,
        groups=8,
        act="silu",
        up_mode="trilinear",
        deep_supervision=False
    ).to(device)

    # Load weights (tolerate checkpoints that store either full dict or state_dict)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("load_state_dict notes:", {"missing": missing, "unexpected": unexpected})
    model.eval()

    # Prepare output folders
    outdir = Path(args.outdir)
    (outdir / "nii").mkdir(parents=True, exist_ok=True)
    (outdir / "viz").mkdir(parents=True, exist_ok=True)

    # ---- Inference: write NIfTIs + optional overlays ----
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device, non_blocking=True)      # [1,1,D,H,W]
            spacing = batch["spacing"][0].cpu().numpy()              # (Dz,Dy,Dx)
            cid = batch["id"][0]

            # Mixed-precision forward; handle (logits, aux) when deep_supervision=True
            with amp.autocast('cuda', dtype=amp_dtype):
                out = model(imgs)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                pred = torch.argmax(logits, dim=1).squeeze(0)        # [D,H,W]
            pred_np = pred.cpu().numpy()

            # Save as NIfTI (.nii.gz)
            out_nii = outdir / "nii" / f"{cid}_pred.nii.gz"
            save_nii_mask(pred_np, spacing, out_nii)
            print(f"Saved: {out_nii}")

            # Save quicklook overlays for the first N cases
            if count < args.viz:
                vol = batch["image"][0, 0].cpu().numpy()             # [D,H,W]
                gt  = batch["label"][0, 0].cpu().numpy().astype(int) # [D,H,W]

                # Choose a slice with GT if possible, else use pred-based heuristic
                z = best_slice(gt if gt.max() > 0 else pred_np)

                sl = window_img(vol[z])
                gt_sl = gt[z]
                pr_sl = pred_np[z]

                # --- 3-panel: image+GT, image+Pred, contours+slice Dice ---
                plt.figure(figsize=(12, 3.4))
                ax1 = plt.subplot(1, 3, 1); ax1.imshow(sl, cmap="gray"); ax1.axis("off"); ax1.set_title(f"{cid} | GT (z={z})")
                ax1.imshow(np.ma.masked_where(gt_sl == 0, gt_sl), alpha=0.45, cmap="tab20")

                ax2 = plt.subplot(1, 3, 2); ax2.imshow(sl, cmap="gray"); ax2.axis("off"); ax2.set_title("Prediction")
                ax2.imshow(np.ma.masked_where(pr_sl == 0, pr_sl), alpha=0.45, cmap="tab20")

                # Quick Dice on that slice (foreground only)
                pr_flat = pr_sl.reshape(-1)
                gt_flat = gt_sl.reshape(-1)
                nonbg = gt_flat > 0
                if nonbg.any():
                    inter = np.sum((pr_flat == gt_flat) & nonbg)
                    den = np.sum(nonbg) + np.sum(pr_flat > 0)
                    slice_dice = (2.0 * inter) / (den + 1e-6)
                else:
                    slice_dice = 0.0

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

                # Simpler two-panel overlay (GT vs Pred)
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

    # ---- Dataset-level Dice check (TEST set) ----
    report = dice_report(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=args.num_classes,
        amp_dtype=amp_dtype,
        label_names={0: "class 0", 1: "class 1", 2: "class 2", 3: "class 3", 4: "class 4", 5: "class 5"},
        threshold=args.threshold
    )
    print("Done. Pass requirement (TEST):", report["pass"])

if __name__ == "__main__":
    main()
