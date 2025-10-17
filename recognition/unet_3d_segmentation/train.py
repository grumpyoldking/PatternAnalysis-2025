# train.py
from pathlib import Path
from datasets import make_loaders_for_hipmri
import matplotlib.pyplot as plt
import numpy as np

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

    # ---- visualize one slice: without and with overlay ----
    # ---- visualize one slice: image only vs. image + label ----

    sample = next(iter(val_loader))                   # full volume (padded)
    img = sample["image"][0, 0].cpu().numpy()         # [D,H,W]
    lbl = sample["label"][0, 0].cpu().numpy().astype(int)  # [D,H,W]
    case_id = sample["id"][0]

    # 1) Inspect label contents
    uniq, counts = np.unique(lbl, return_counts=True)
    print("Case:", case_id, "| unique label IDs (value: count) ->",
        {int(u): int(c) for u, c in zip(uniq, counts)})

    # 2) Pick a slice with the most label (fallback: middle)
    label_area_per_slice = (lbl > 0).reshape(lbl.shape[0], -1).sum(axis=1)
    if label_area_per_slice.max() > 0:
        z = int(label_area_per_slice.argmax())
    else:
        z = img.shape[0] // 2
        print("⚠️ No nonzero labels found in this volume; showing middle slice.")

    # Optional: simple intensity windowing for better contrast
    sl = img[z]
    p2, p98 = np.percentile(sl, (2, 98))
    sl_disp = np.clip((sl - p2) / (p98 - p2 + 1e-6), 0, 1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Case: {case_id}  |  slice z={z}", y=1.03)

    # Left: image only
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(sl_disp, cmap="gray")
    ax1.set_title("Image (no overlay)")
    ax1.axis("off")

    # Right: image + label overlay (both filled + contour)
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(sl_disp, cmap="gray")
    # show filled overlay (mask out background=0)
    lbl_slice = lbl[z]
    lbl_masked = np.ma.masked_where(lbl_slice == 0, lbl_slice)
    ax2.imshow(lbl_masked, alpha=0.45, interpolation="nearest", cmap="tab20")
    # add crisp boundaries for visibility
    vals = [v for v in np.unique(lbl_slice) if v != 0]
    if vals:
        ax2.contour(lbl_slice, levels=vals, linewidths=1.0)
    ax2.set_title("Image + Label overlay")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
