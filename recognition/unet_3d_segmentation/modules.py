from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import time

import torch
import torch.nn as nn
# keep using CUDA AMP's autocast; it does NOT accept device_type=
from torch import amp

# -------------------- dataset discovery --------------------

def find_dataset_root() -> str:
    """
    Resolve dataset root that contains BOTH:
      semantic_MRs_anon/ and semantic_labels_anon/
    Priority:
      1) $DATASET_ROOT (if set)
      2) Google Drive path you showed in Colab
         /content/drive/MyDrive/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data
      3) A recursive search under /content/drive/MyDrive
      4) Original fallback: scan repo tree near this file
    """
    # 1) Environment override
    env_root = os.getenv("DATASET_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "semantic_MRs_anon").is_dir() and (p / "semantic_labels_anon").is_dir():
            return str(p)

    # 2) Your Google Drive dataset path
    gd_specific = Path("/content/drive/MyDrive/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data")
    if (gd_specific / "semantic_MRs_anon").is_dir() and (gd_specific / "semantic_labels_anon").is_dir():
        return str(gd_specific)

    # 3) Try to find it anywhere under MyDrive (shallow recursive search)
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.exists():
        for cand in mydrive.rglob("*"):
            if not cand.is_dir():
                continue
            img_dir = cand / "semantic_MRs_anon"
            lbl_dir = cand / "semantic_labels_anon"
            if img_dir.is_dir() and lbl_dir.is_dir():
                return str(cand)

    # 4) Original local fallback (repo tree)
    here = Path(__file__).resolve().parent
    for cand in [here, *here.rglob("*")]:
        if not cand.is_dir():
            continue
        img_dir = cand / "semantic_MRs_anon"
        lbl_dir = cand / "semantic_labels_anon"
        if img_dir.is_dir() and lbl_dir.is_dir():
            return str(cand)

    raise FileNotFoundError(
        "Could not find dataset root. Tried:\n"
        f"  $DATASET_ROOT={env_root}\n"
        f"  {gd_specific}\n"
        f"  Under {mydrive} (recursive)\n"
        f"  Near this script: {here}\n"
        "Make sure Google Drive is mounted and your folders are named "
        "'semantic_MRs_anon' and 'semantic_labels_anon'.\n"
        "Alternatively, set:  os.environ['DATASET_ROOT'] = '<absolute_path>'"
    )

# -------------------- (helpers) --------------------
def case_root(case_id: str) -> str:
    m = re.match(r"^(Case_\d+)", case_id)
    if m:
        return m.group(1)
    parts = case_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else case_id

def pick_best_slice(img_dhw: np.ndarray, lbl_dhw: np.ndarray) -> int:
    area = (lbl_dhw > 0).reshape(lbl_dhw.shape[0], -1).sum(axis=1)
    return int(area.argmax()) if area.max() > 0 else img_dhw.shape[0] // 2

def window_img(x: np.ndarray):
    p2, p98 = np.percentile(x, (2, 98))
    return np.clip((x - p2) / (p98 - p2 + 1e-6), 0, 1)

def visualize_5_unique_cases(val_loader, save_path: Path | None = None):
    unique_samples = []
    seen_roots = set()
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
        axes = np.array([axes])

    for row, sample in enumerate(unique_samples):
        img = sample["image"][0, 0].cpu().numpy()               # [D,H,W]
        lbl = sample["label"][0, 0].cpu().numpy().astype(int)   # [D,H,W]
        case_id = sample["id"][0]
        root_id = case_root(case_id)

        uniq, counts = np.unique(lbl, return_counts=True)
        print(f"[{row+1}/{n}] {root_id} ({case_id}) | labels ->",
              {int(u): int(c) for u, c in zip(uniq, counts)})

        z = pick_best_slice(img, lbl)
        sl_disp = window_img(img[z])
        lbl_slice = lbl[z]

        ax1 = axes[row, 0]
        ax1.imshow(sl_disp, cmap="gray")
        ax1.set_title(f"{root_id} | z={z} (no overlay)", fontsize=9)
        ax1.axis("off")

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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved viz to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

# -------------------- losses & metrics --------------------

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, ignore_background: bool = False):
        super().__init__()
        self.eps = eps
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        N, C = probs.shape[:2]
        one_hot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1)

        start_c = 1 if self.ignore_background and C > 1 else 0
        dims = (0, 2, 3, 4)
        inter = (probs[:, start_c:] * one_hot[:, start_c:]).sum(dim=dims)
        den   = probs[:, start_c:].sum(dim=dims) + one_hot[:, start_c:].sum(dim=dims)
        dice = (2 * inter + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes: int):
    model.eval()
    dices_sum = torch.zeros(num_classes, device=device)
    dices_cnt = torch.zeros(num_classes, device=device)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(ignore_background=False)

    tot_ce = 0.0
    tot_dice = 0.0
    n_batches = 0

    for batch in val_loader:
        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"][:, 0].long().to(device, non_blocking=True)

        logits = model(imgs)
        tot_ce += ce_loss(logits, labels).item()
        tot_dice += dice_loss(logits, labels).item()
        n_batches += 1

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

    mean_ce = tot_ce / max(n_batches, 1)
    mean_dice_loss = tot_dice / max(n_batches, 1)
    per_class_dice = torch.where(dices_cnt > 0, dices_sum / dices_cnt.clamp_min(1), torch.zeros_like(dices_sum))
    mean_dice = per_class_dice[1:].mean().item() if num_classes > 1 else per_class_dice.mean().item()

    return {
        "val_ce": mean_ce,
        "val_dice_loss": mean_dice_loss,
        "val_per_class_dice": per_class_dice.tolist(),
        "val_mean_dice_excl_bg": mean_dice
    }

# -------------------- Improved UNet (3D) --------------------
# Residual blocks + SCSE attention + learnable down/upsampling + optional deep supervision
# Keeps interface: forward(x) -> logits [B, C, D, H, W]

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
    """Concurrent Spatial & Channel 'Squeeze & Excitation' (3D)."""
    def __init__(self, ch, r=16):
        super().__init__()
        red = max(ch // r, 1)
        # Channel SE
        self.cse_avg = nn.AdaptiveAvgPool3d(1)
        self.cse_fc = nn.Sequential(
            nn.Conv3d(ch, red, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(red, ch, 1, bias=True),
            nn.Sigmoid(),
        )
        # Spatial SE
        self.sse = nn.Sequential(nn.Conv3d(ch, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        c = self.cse_fc(self.cse_avg(x))
        s = self.sse(x)
        return x * c + x * s


class AttGate3d(nn.Module):
    """Attention gate for skip connections (U-Net)."""
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
    """Learnable downsampling (strided conv) + residual block + SCSE."""
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
    """Upsample + attention-gated skip + residual fuse + SCSE."""
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
    """
    Encoder-decoder with residual blocks, SCSE attention, attention-gated skips,
    learnable downsampling, and optional deep supervision.

    Args:
        in_channels: input channels (e.g., 1)
        out_channels: number of classes (logits)
        features: tuple of channel widths per stage, length >= 4 recommended
        dropout: dropout rate within residual blocks
        groups: GroupNorm groups
        act: 'silu' (SiLU) | 'relu' | 'lrelu'
        up_mode: 'trilinear' | 'deconv'
        deep_supervision: if True, returns (logits, aux_list); default False -> returns logits only
    """
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

        # Stem
        self.stem = ResBlock3d(in_channels, chans[0], dropout=dropout, groups=groups, act=act)
        self.stem_scse = SCSE3d(chans[0])

        # Encoder
        self.enc = nn.ModuleList()
        for i in range(len(chans) - 1):
            self.enc.append(DownBlock3d(chans[i], chans[i+1], dropout=dropout, groups=groups, act=act))

        # Decoder
        self.dec = nn.ModuleList()
        for i in reversed(range(len(chans) - 1)):
            in_ch  = chans[i+1]
            skip_ch= chans[i]
            out_ch = chans[i]
            self.dec.append(UpBlock3d(in_ch, skip_ch, out_ch, dropout=dropout, groups=groups, act=act, up_mode=up_mode))

        # Heads
        self.head = nn.Conv3d(chans[0], out_channels, 1, bias=True)

        if deep_supervision:
            # Create aux heads for deeper decoder levels (except final)
            self.aux_heads = nn.ModuleList([
                nn.Conv3d(chans[i], out_channels, 1, bias=True) for i in range(1, len(chans)-0)
            ])

    def forward(self, x):
        # Encoder path
        s0 = self.stem_scse(self.stem(x))
        feats = [s0]
        y = s0
        for down in self.enc:
            y = down(y)
            feats.append(y)

        # Decoder path
        aux = []
        for i, up in enumerate(self.dec):
            skip = feats[-(i+2)]  # go backwards over encoder+stem features
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

# -------------------- main (CUDA + training loop) --------------------

def main():
    # --- CUDA / Colab setup ---
    assert torch.cuda.is_available(), "CUDA GPU not found. In Colab: Runtime → Change runtime type → GPU."
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f"Using GPU: {gpu_name} (CC {cc_major}.{cc_minor})")

    # AMP dtype: bf16 on A100/H100, else fp16
    amp_dtype = torch.bfloat16 if (cc_major >= 8) else torch.float16
    print(f"AMP dtype: {amp_dtype}")

    # --- Data ---
    root = find_dataset_root()
    print("Dataset root:", root)

    # NOTE: make_loaders_for_hipmri must be defined elsewhere in your project.
    train_loader, val_loader = make_loaders_for_hipmri(
        root=root,
        target_spacing=(2.0, 2.0, 2.0),
        patch_size=(128, 128, 64),
        batch_size=2,
        workers=2
    )

    b = next(iter(train_loader))
    print("Train batch:", b["image"].shape, b["label"].shape, b["id"][:2])

    visualize_5_unique_cases(val_loader, save_path=Path("runs/preview_val_cases.png"))

    # --- Model / Optimizer / Loss ---
    num_classes = 6  # <-- set for your dataset
    model = ImprovedUNet3D(
        in_channels=1,
        out_channels=num_classes,
        features=(32, 64, 128, 256, 512),   # reduce if VRAM is tight
        dropout=0.1,
        groups=8,
        act="silu",
        up_mode="trilinear",                # or "deconv"
        deep_supervision=False              # keep False to avoid changing your loss loop
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(ignore_background=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = amp.GradScaler('cuda', enabled=True)

    # --- Training config ---
    epochs =  15
    grad_clip = 1.0
    save_dir = Path("runs/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_ce = 0.0
        epoch_dice = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)          # [B,1,D,H,W]
            labels = batch["label"][:, 0].long().to(device, non_blocking=True)  # [B,D,H,W]

            optimizer.zero_grad(set_to_none=True)

            # PATCH: use CUDA AMP's autocast WITHOUT device_type=
            with amp.autocast('cuda', dtype=amp_dtype):
                logits = model(imgs)
                loss_ce = ce_loss(logits, labels)
                loss_dice = dice_loss(logits, labels)
                loss = 0.5 * loss_ce + 0.5 * loss_dice

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_ce += loss_ce.item()
            epoch_dice += loss_dice.item()
            n_batches += 1

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

        if metrics["val_mean_dice_excl_bg"] > best_dice:
            best_dice = metrics["val_mean_dice_excl_bg"]
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  ↳ New best Dice (excl bg): {best_dice:.4f} — saved to runs/checkpoints/best.pt")

    print("Training complete. Best Dice (excl bg):", best_dice)

if __name__ == "__main__":
    main()
