# train.py
from pathlib import Path
from datasets import make_loaders_for_hipmri
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
import re

def find_dataset_root() -> str:
    here = Path(__file__).resolve().parent  # .../unet_3d_segmentation
    for cand in [here, *here.rglob("*")]:
        if not cand.is_dir():
            continue
        img_dir = cand / "semantic_MRs_anon"
        lbl_dir = cand / "semantic_labels_anon"
        if img_dir.is_dir() and lbl_dir.is_dir():
            return str(cand)
    raise FileNotFoundError(
        "Could not find a folder containing both 'semantic_MRs_anon' and "
        "'semantic_labels_anon' under: " + str(here)
    )

def case_root(case_id: str) -> str:
    """
    Extract 'Case_<num>' from ids like:
      'Case_004_Week0_SEMANTIC_LFOV' -> 'Case_004'
      'Case_010_Week3_T2W'           -> 'Case_010'
    Fall back to the first two underscore chunks if regex fails.
    """
    m = re.match(r"^(Case_\d+)", case_id)
    if m:
        return m.group(1)
    parts = case_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else case_id

def pick_best_slice(img_dhw: np.ndarray, lbl_dhw: np.ndarray) -> int:
    """Choose the axial slice index with the most labeled voxels (fallback: middle)."""
    label_area_per_slice = (lbl_dhw > 0).reshape(lbl_dhw.shape[0], -1).sum(axis=1)
    if label_area_per_slice.max() > 0:
        return int(label_area_per_slice.argmax())
    return img_dhw.shape[0] // 2

def window_img(x: np.ndarray):
    """Simple percentile windowing for display."""
    p2, p98 = np.percentile(x, (2, 98))
    return np.clip((x - p2) / (p98 - p2 + 1e-6), 0, 1)

def main():
    root = find_dataset_root()
    print("Using dataset root:", root)

    train_loader, val_loader = make_loaders_for_hipmri(
        root=root,
        target_spacing=(2.0, 2.0, 2.0),
        patch_size=(128, 128, 64),
        batch_size=2,
        workers=0
    )

    # quick smoke test
    b = next(iter(train_loader))
    print("Train batch:", b["image"].shape, b["label"].shape, b["id"][:2])

    # ---- collect 5 unique cases (Case_<id>), ignoring Week ----
    unique_samples = []
    seen_roots = set()
    MAX_SCAN = 1000  # safety cap to avoid endless iteration on small datasets
    scanned = 0
    for sample in val_loader:
        scanned += 1
        cid_full = sample["id"][0]
        root_id = case_root(cid_full)
        if root_id in seen_roots:
            if scanned >= MAX_SCAN:
                break
            continue
        seen_roots.add(root_id)
        unique_samples.append(sample)
        if len(unique_samples) == 5:
            break
        if scanned >= MAX_SCAN:
            break

    n = len(unique_samples)
    if n == 0:
        print("No validation samples found.")
        return
    if n < 5:
        print(f"Only found {n} unique cases in validation set.")

    # ---- visualize the selected unique cases (n rows × 2 cols) ----
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 2.4 * n))
    if n == 1:
        axes = np.array([axes])  # make it 2D indexable

    for row, sample in enumerate(unique_samples):
        img = sample["image"][0, 0].cpu().numpy()               # [D,H,W]
        lbl = sample["label"][0, 0].cpu().numpy().astype(int)   # [D,H,W]
        case_id = sample["id"][0]
        root_id = case_root(case_id)

        # print label stats
        uniq, counts = np.unique(lbl, return_counts=True)
        print(f"[{row+1}/{n}] {root_id} ({case_id}) | labels ->",
              {int(u): int(c) for u, c in zip(uniq, counts)})

        z = pick_best_slice(img, lbl)
        sl_disp = window_img(img[z])
        lbl_slice = lbl[z]

        # left: image only
        ax1 = axes[row, 0]
        ax1.imshow(sl_disp, cmap="gray")
        ax1.set_title(f"{root_id} | z={z} (no overlay)", fontsize=9)
        ax1.axis("off")

        # right: image + label overlay
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
    plt.show()

if __name__ == "__main__":
    main()
