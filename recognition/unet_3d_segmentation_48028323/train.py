from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import time

import torch
import torch.nn as nn
from torch import amp

# NOTE: this script assumes the following are available elsewhere in your project:
#   from dataloaders.prostate3d import make_loaders_for_hipmri
#   from models.improved_unet3d import ImprovedUNet3D
# Import them at the top of your actual file:
# from dataloaders.prostate3d import make_loaders_for_hipmri
# from models.improved_unet3d import ImprovedUNet3D

# -------------------- dataset discovery --------------------

def find_dataset_root() -> str:
    """
    Try multiple locations to find a dataset root that contains both:
      - semantic_MRs_anon/
      - semantic_labels_anon/
    Priority order:
      1) $DATASET_ROOT (explicit, preferred)
      2) Known Google Drive path used in Colab
      3) Recursive search under /content/drive/MyDrive
      4) Search near this script in the repo
    """
    # 1) Environment override for reproducibility and CI-friendly usage
    env_root = os.getenv("DATASET_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "semantic_MRs_anon").is_dir() and (p / "semantic_labels_anon").is_dir():
            return str(p)

    # 2) Specific Google Drive path that you used previously in Colab
    gd_specific = Path("/content/drive/MyDrive/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data")
    if (gd_specific / "semantic_MRs_anon").is_dir() and (gd_specific / "semantic_labels_anon").is_dir():
        return str(gd_specific)

    # 3) Broader search within Drive (can be slow)
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.exists():
        for cand in mydrive.rglob("*"):
            if not cand.is_dir():
                continue
            img_dir = cand / "semantic_MRs_anon"
            lbl_dir = cand / "semantic_labels_anon"
            if img_dir.is_dir() and lbl_dir.is_dir():
                return str(cand)

    # 4) Last resort: search relative to this file (local dev)
    here = Path(__file__).resolve().parent
    for cand in [here, *here.rglob("*")]:
        if not cand.is_dir():
            continue
        img_dir = cand / "semantic_MRs_anon"
        lbl_dir = cand / "semantic_labels_anon"
        if img_dir.is_dir() and lbl_dir.is_dir():
            return str(cand)

    # If we got here, nothing matched — fail loudly with guidance
    raise FileNotFoundError("Dataset root not found; set DATASET_ROOT or mount Drive with expected folders.")

# -------------------- (helpers) --------------------
def case_root(case_id: str) -> str:
    """
    Collapse a Case_XXX_WeekY id down to its case root 'Case_XXX' for grouping.
    If the pattern is unusual, fall back to the first two underscore-separated tokens.
    """
    m = re.match(r"^(Case_\d+)", case_id)
    if m:
        return m.group(1)
    parts = case_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else case_id

def pick_best_slice(img_dhw: np.ndarray, lbl_dhw: np.ndarray) -> int:
    """
    Heuristic to pick a representative axial slice index:
    choose the slice with the maximal number of labeled voxels;
    if no labels exist, choose the middle slice.
    """
    area = (lbl_dhw > 0).reshape(lbl_dhw.shape[0], -1).sum(axis=1)
    return int(area.argmax()) if area.max() > 0 else img_dhw.shape[0] // 2

def window_img(x: np.ndarray):
    """
    Simple intensity windowing for display:
    map [2nd, 98th] percentiles to [0,1] and clamp.
    """
    p2, p98 = np.percentile(x, (2, 98))
    return np.clip((x - p2) / (p98 - p2 + 1e-6), 0, 1)

def visualize_5_unique_cases(val_loader, save_path: Path | None = None):
    """
    Fetch up to 5 unique case roots from the validation loader and visualize:
      - left: grayscale slice
      - right: same slice with label overlay and contour
    If save_path is provided, save the figure; otherwise, show it.
    """
    unique_samples = []
    seen_roots = set()

    # Collect first 5 distinct case roots
    for sample in val_loader:
        cid_full = sample["id"][0]
        root_id = case_root(cid_full)
        if root_id in seen_roots:
            continue
        seen_roots.add(root_id)
        unique_samples.append(sample)
        if len(unique_samples) == 5:
            break

    n = len(unique_samples)
    if n == 0:
        print("No validation samples found for visualization.")
        return
    if n < 5:
        print(f"Only found {n} unique cases in validation set.")

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 2.4 * n))
    if n == 1:
        axes = np.array([axes])  # normalize to 2D array for indexing

    for row, sample in enumerate(unique_samples):
        # Each 'sample' is a collated batch of size 1 from the val loader
        img = sample["image"][0, 0].cpu().numpy()               # [D,H,W]
        lbl = sample["label"][0, 0].cpu().numpy().astype(int)   # [D,H,W]
        case_id = sample["id"][0]
        root_id = case_root(case_id)

        # Print class histogram info for the chosen volume
        uniq, counts = np.unique(lbl, return_counts=True)
        print(f"[{row+1}/{n}] {root_id} ({case_id}) | labels ->",
              {int(u): int(c) for u, c in zip(uniq, counts)})

        # Choose slice and prep overlays
        z = pick_best_slice(img, lbl)
        sl_disp = window_img(img[z])
        lbl_slice = lbl[z]

        # Left: grayscale
        ax1 = axes[row, 0]
        ax1.imshow(sl_disp, cmap="gray")
        ax1.set_title(f"{root_id} | z={z} (no overlay)", fontsize=9)
        ax1.axis("off")

        # Right: grayscale + colored labels + contour
        ax2 = axes[row, 1]
        ax2.imshow(sl_disp, cmap="gray")
        lbl_masked = np.ma.masked_where(lbl_slice == 0, lbl_slice)
        ax2.imshow(lbl_masked, alpha=0.45, interpolation="nearest", cmap="tab20")
        vals = [v for v in np.unique(lbl_slice) if v != 0]
        if vals:
            ax2.contour(lbl_slice, levels=vals, linewidths=1.0)
        ax2.set_title(f"{root_id} | Image + Label", fontsize=9)
        ax2.axis("off")

    plt.tight_layout()
    if save_path is not None:
        # Save to disk (create parent dir if needed)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved viz to: {save_path}")
        plt.close(fig)
    else:
        # Display inline (e.g., in notebooks)
        plt.show()

# -------------------- losses & metrics --------------------

class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss.
    Optionally ignore background channel when computing the mean (set ignore_background=True).
    """
    def __init__(self, eps: float = 1e-6, ignore_background: bool = False):
        super().__init__()
        self.eps = eps
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
          logits: [B, C, D, H, W] unnormalized scores
          target: [B, D, H, W] integer labels in [0..C-1]
        Returns:
          scalar Dice loss
        """
        probs = torch.softmax(logits, dim=1)            # [B,C,D,H,W]
        N, C = probs.shape[:2]
        # One-hot encode targets to [B,C,D,H,W]
        one_hot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1)

        # Optionally drop background for loss averaging
        start_c = 1 if self.ignore_background and C > 1 else 0
        dims = (0, 2, 3, 4)                              # sum over batch + spatial
        inter = (probs[:, start_c:] * one_hot[:, start_c:]).sum(dim=dims)
        den   = probs[:, start_c:].sum(dim=dims) + one_hot[:, start_c:].sum(dim=dims)
        dice = (2 * inter + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes: int):
    """
    Evaluation loop:
      - Computes mean CE and DiceLoss on val set
      - Reports per-class Dice (including background) and mean Dice excluding background
    """
    model.eval()
    dices_sum = torch.zeros(num_classes, device=device)   # accumulate per-class dice
    dices_cnt = torch.zeros(num_classes, device=device)   # count of batches with class present
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(ignore_background=False)

    tot_ce = 0.0
    tot_dice = 0.0
    n_batches = 0

    for batch in val_loader:
        imgs = batch["image"].to(device, non_blocking=True)         # [B,1,D,H,W]
        labels = batch["label"][:, 0].long().to(device, non_blocking=True)  # [B,D,H,W]

        # Forward pass
        logits = model(imgs)

        # Accumulate scalar losses
        tot_ce += ce_loss(logits, labels).item()
        tot_dice += dice_loss(logits, labels).item()
        n_batches += 1

        # Hard prediction for Dice reporting (argmax over classes)
        pred = torch.argmax(logits, dim=1)  # [B,D,H,W]
        for c in range(num_classes):
            p = (pred == c).float()
            t = (labels == c).float()
            inter = (p * t).sum()
            den = p.sum() + t.sum()
            if den > 0:
                dice_c = (2 * inter) / (den + 1e-6)
                dices_sum[c] += dice_c
                dices_cnt[c] += 1

    # Mean over batches
    mean_ce = tot_ce / max(n_batches, 1)
    mean_dice_loss = tot_dice / max(n_batches, 1)
    # Average only over batches where the class appears (avoid div by 0)
    per_class_dice = torch.where(dices_cnt > 0, dices_sum / dices_cnt.clamp_min(1), torch.zeros_like(dices_sum))
    # Commonly report mean dice across foreground classes only
    mean_dice = per_class_dice[1:].mean().item() if num_classes > 1 else per_class_dice.mean().item()

    return {
        "val_ce": mean_ce,
        "val_dice_loss": mean_dice_loss,
        "val_per_class_dice": per_class_dice.tolist(),
        "val_mean_dice_excl_bg": mean_dice
    }

# -------------------- main (CUDA + training loop) --------------------

def main():
    """
    Full training entry point:
      - CUDA/AMP initialization
      - Data loaders
      - Model/optimizer/loss setup
      - Train/validate loop with checkpointing
    """
    # Require a CUDA device for 3D volumes (training will be slow on CPU)
    assert torch.cuda.is_available(), "CUDA GPU not found."
    device = torch.device("cuda")

    # Speed heuristics for convnets
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")  # PyTorch 2.0+ optional
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True     # allow TF32 on Ampere+ for speed

    # Report device info and choose AMP dtype
    gpu_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f"Using GPU: {gpu_name} (CC {cc_major}.{cc_minor})")

    # bf16 is best on Ampere+; otherwise fall back to fp16
    amp_dtype = torch.bfloat16 if (cc_major >= 8) else torch.float16
    print(f"AMP dtype: {amp_dtype}")

    # --- Data ---
    root = find_dataset_root()
    print("Dataset root:", root)

    # Build train/val loaders (HIP-MRI folder names + fixed spacing)
    train_loader, val_loader = make_loaders_for_hipmri(
        root=root,
        target_spacing=(2.0, 2.0, 2.0),
        patch_size=(128, 128, 64),
        batch_size=2,
        workers=2
    )

    # Peek at a single batch to verify shapes
    b = next(iter(train_loader))
    print("Train batch:", b["image"].shape, b["label"].shape, b["id"][:2])

    # Quick qualitative sanity check: save a small panel of val cases
    visualize_5_unique_cases(val_loader, save_path=Path("runs/preview_val_cases.png"))

    # --- Model / Optimizer / Loss ---
    num_classes = 6  # <-- adjust to your dataset's label count (incl. background class 0)
    model = ImprovedUNet3D(
        in_channels=1,
        out_channels=num_classes,
        features=(32, 64, 128, 256, 512),  # reduce if VRAM is tight
        dropout=0.1,
        groups=8,
        act="silu",
        up_mode="trilinear",
        deep_supervision=False             # keep False to avoid changing the loss loop
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(ignore_background=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = amp.GradScaler('cuda', enabled=True)  # gradient scaling for mixed precision

    # --- Training config ---
    epochs = 15
    grad_clip = 1.0
    save_dir = Path("runs/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0  # track best validation Dice (excluding background)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_ce = 0.0
        epoch_dice = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)                 # [B,1,D,H,W]
            labels = batch["label"][:, 0].long().to(device, non_blocking=True)  # [B,D,H,W]

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward/backward
            with amp.autocast('cuda', dtype=amp_dtype):
                logits = model(imgs)
                loss_ce = ce_loss(logits, labels)
                loss_dice = dice_loss(logits, labels)
                loss = 0.5 * loss_ce + 0.5 * loss_dice  # simple balanced combo

            # Backprop with scaling to prevent underflow
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)  # unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Bookkeeping
            epoch_ce += loss_ce.item()
            epoch_dice += loss_dice.item()
            n_batches += 1

        # Scheduler step per epoch (CosineAnnealing)
        scheduler.step()
        dt = time.time() - t0
        train_ce = epoch_ce / max(n_batches, 1)
        train_dice_loss = epoch_dice / max(n_batches, 1)

        # --- Validate ---
        metrics = evaluate(model, val_loader, device, num_classes)
        msg = (f"Epoch {epoch:03d} | {dt:5.1f}s | "
               f"train CE {train_ce:.4f} | train DiceLoss {train_dice_loss:.4f} | "
               f"val CE {metrics['val_ce']:.4f} | val DiceLoss {metrics['val_dice_loss']:.4f} | "
               f"val mean Dice excl bg {metrics['val_mean_dice_excl_bg']:.4f}")
        print(msg)

        # --- Checkpointing ---
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "num_classes": num_classes,
        }
        torch.save(ckpt, save_dir / "last.pt")

        # Track the best model by foreground mean Dice
        if metrics["val_mean_dice_excl_bg"] > best_dice:
            best_dice = metrics["val_mean_dice_excl_bg"]
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  ↳ New best Dice (excl bg): {best_dice:.4f} — saved to runs/checkpoints/best.pt")

    print("Training complete. Best Dice (excl bg):", best_dice)


if __name__ == "__main__":
    main()
