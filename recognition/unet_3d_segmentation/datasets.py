# dataloaders/prostate3d.py
# deps: pip/conda install torch nibabel scipy numpy
# Purpose:
#   3D prostate dataset utilities for training/evaluating volumetric segmentation models.
#   - Robust image/label pairing via canonical keys (Case_<id>_Week<k>)
#   - NIfTI I/O with canonical orientation and spacing alignment
#   - Resampling to target voxel spacing
#   - Foreground cropping / random patch sampling / padding to model multiples
#   - Light data augmentation (flips, small rotations, gamma-like intensity, noise)
#   - Train/val/test DataLoaders with safe collate for [B,1,D,H,W] tensors
#
# All arrays are handled in array order (D, H, W) == (Z, Y, X).
# Spacing is consistently reported as (Dz, Dy, Dx) to match these axes.

import os, glob, random, re
from typing import Tuple, Optional, List, Dict
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage as ndi

# -------------------------- Auto-discover dataset root --------------------------

def find_dataset_root() -> str:
    """
    Find a directory that contains BOTH 'semantic_MRs_anon' and 'semantic_labels_anon'.
    Priority:
      1) $DATASET_ROOT (if set and valid)
      2) Known Google Drive path used in Colab
         /content/drive/MyDrive/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data
      3) Recursive search under /content/drive/MyDrive
      4) Recursive search near this file
    """
    from pathlib import Path

    def has_dirs(p: Path) -> bool:
        return (p / "semantic_MRs_anon").is_dir() and (p / "semantic_labels_anon").is_dir()

    # 1) Environment override
    env_root = os.getenv("DATASET_ROOT")
    if env_root and has_dirs(Path(env_root)):
        return env_root

    # 2) Known Colab path
    gd = Path("/content/drive/MyDrive/Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data")
    if has_dirs(gd):
        return str(gd)

    # 3) Search MyDrive (shallow recursive)
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.exists():
        for cand in mydrive.rglob("*"):
            if cand.is_dir() and has_dirs(cand):
                return str(cand)

    # 4) Search near this file
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for cand in [here, *here.rglob("*")]:
        if cand.is_dir() and has_dirs(cand):
            return str(cand)

    raise FileNotFoundError(
        "Could not find dataset root with 'semantic_MRs_anon' and 'semantic_labels_anon'. "
        "Set DATASET_ROOT or mount Google Drive."
    )

# -------------------------- helpers --------------------------

def load_nii(path: str):
    """
    Load a NIfTI and reorient to canonical RAS so that header zooms align with axes.

    Returns
    -------
    data : np.ndarray (float32, shape [D,H,W])
        Volume data in array order (Z,Y,X) == (D,H,W).
    space : tuple (Dz, Dy, Dx)
        Voxel spacing in mm aligned to (D,H,W) order.
    """
    nimg = nib.load(path)
    nimg = nib.as_closest_canonical(nimg)          # enforce RAS orientation
    data = nimg.get_fdata(dtype=np.float32)        # (X, Y, Z) in RAS
    data = np.transpose(data, (2, 1, 0))           # -> (Z, Y, X) == (D, H, W)
    sx, sy, sz = nimg.header.get_zooms()[:3]       # spacings along X,Y,Z
    spacing = (sz, sy, sx)                         # -> (Dz, Dy, Dx) for (D,H,W)
    return data, spacing


def resample_to_spacing(
    img: np.ndarray,
    spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int
) -> np.ndarray:
    """
    Resample 3D array (D,H,W) from given 'spacing' to 'target_spacing'.
    order: 1 = linear (images), 0 = nearest (labels).
    """
    zoom = tuple(s / t for s, t in zip(spacing, target_spacing))  # per-axis scale
    return ndi.zoom(img, zoom=zoom, order=order)


def percentile_clip_zscore(x: np.ndarray, pmin=0.5, pmax=99.5, eps=1e-6) -> np.ndarray:
    """
    Robust normalize image intensities:
    1) Clip to [pmin, pmax] percentiles
    2) Z-score normalize
    """
    lo, hi = np.percentile(x, [pmin, pmax])
    x = np.clip(x, lo, hi)
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + eps)


def compute_bbox(mask: np.ndarray, pad: Tuple[int,int,int]=(8,8,8)) -> Optional[Tuple[slice,slice,slice]]:
    """Return a padded bounding box around nonzero mask; None if empty."""
    inds = np.where(mask > 0)
    if len(inds[0]) == 0:
        return None
    zmin, zmax = inds[0].min(), inds[0].max()
    ymin, ymax = inds[1].min(), inds[1].max()
    xmin, xmax = inds[2].min(), inds[2].max()
    pz, py, px = pad
    zmin, zmax = max(0, zmin - pz), min(mask.shape[0]-1, zmax + pz)
    ymin, ymax = max(0, ymin - py), min(mask.shape[1]-1, ymax + py)
    xmin, xmax = max(0, xmin - px), min(mask.shape[2]-1, xmax + px)
    return (slice(zmin, zmax+1), slice(ymin, ymax+1), slice(xmin, xmax+1))


def random_crop_3d(img, msk, size: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray]:
    """Random crop (D,H,W) to size (dz,dy,dx); symmetric pad if needed."""
    d, h, w = img.shape
    dz, dy, dx = size
    pad_z = max(0, dz - d); pad_y = max(0, dy - h); pad_x = max(0, dx - w)
    if pad_z or pad_y or pad_x:
        pz0, pz1 = pad_z // 2, pad_z - pad_z//2
        py0, py1 = pad_y // 2, pad_y - pad_y//2
        px0, px1 = pad_x // 2, pad_x - pad_x//2
        img = np.pad(img, ((pz0,pz1),(py0,py1),(px0,px1)), mode='constant')
        msk = np.pad(msk, ((pz0,pz1),(py0,py1),(px0,px1)), mode='constant')
        d, h, w = img.shape
    z0 = random.randint(0, max(0, d - dz)) if d > dz else 0
    y0 = random.randint(0, max(0, h - dy)) if h > dy else 0
    x0 = random.randint(0, max(0, w - dx)) if w > dx else 0
    return img[z0:z0+dz, y0:y0+dy, x0:x0+dx], msk[z0:z0+dz, y0:y0+dy, x0:x0+dx]


def pad_to_multiple(arr: np.ndarray, m: int = 16) -> np.ndarray:
    """Pad (D,H,W) so each dim is a multiple of m (pad at the end only)."""
    D, H, W = arr.shape
    pD = (m - D % m) % m
    pH = (m - H % m) % m
    pW = (m - W % m) % m
    return np.pad(arr, ((0, pD), (0, pH), (0, pW)), mode="constant")


def rand_flip3d(img, msk, p=0.5):
    """Random flips along Z, Y, X; keeps image/mask aligned."""
    if random.random() < p:
        img = img[::-1, ...]; msk = msk[::-1, ...]
    if random.random() < p:
        img = img[:, ::-1, :]; msk = msk[:, ::-1, :]
    if random.random() < p:
        img = img[:, :, ::-1]; msk = msk[:, :, ::-1]
    return img, msk


def rand_intensity(img, gamma_range=(0.9, 1.1), noise_std=0.03):
    """Gamma-like contrast + small Gaussian noise, applied to normalized image."""
    g = random.uniform(*gamma_range)
    x = img - img.min() + 1e-6
    x = x ** g
    x = x + (img.min() - x.min())  # recenter roughly
    if noise_std > 0:
        x = x + np.random.normal(0, noise_std, size=x.shape).astype(np.float32)
    return x


def _strip_nii_ext(basename: str) -> str:
    """'file.nii.gz' -> 'file', 'file.nii' -> 'file'."""
    return basename.replace(".nii.gz", "").replace(".nii", "")


def _key_from_name(name: str) -> Optional[str]:
    """
    Extract canonical key 'Case_<id>_Week<k>' from filenames such as:
        Case_004_Week0_SEMANTIC_LFOV.nii.gz
        Case_004_Week0_T2W.nii.gz
        Case_004_Week0.nii.gz
    Returns None if the pattern isn't found.
    """
    m = re.match(r"^(Case_\d+_Week\d+)", _strip_nii_ext(name))
    return m.group(1) if m else None


def _case_root_from_key(key: str) -> Optional[str]:
    """'Case_004_Week3' -> 'Case_004' for grouping by patient/case."""
    m = re.match(r"^(Case_\d+)", key)
    return m.group(1) if m else None

# -------------------------- Dataset --------------------------

class Prostate3DDataset(Dataset):
    """
    Directory layout:
      root/
        images/ or imagesTr/   (images with names like Case_004_Week0_*.nii.gz)
        labels/ or labelsTr/   (labels with names like Case_004_Week0_*SEMANTIC*.nii.gz)
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dirnames: Tuple[str,str] = ("images", "imagesTr"),
        lbl_dirnames: Tuple[str,str] = ("labels", "labelsTr"),
        target_spacing: Optional[Tuple[float,float,float]] = None,  # e.g., (3.0, 1.0, 1.0) for (D,H,W)
        crop_to_foreground: bool = True,
        patch_size: Optional[Tuple[int,int,int]] = (128,128,64),
        samples_per_vol: int = 1,      # used when patch_size is not None
        augment: bool = True,
        for_eval_full_volume: bool = False,
        fg_crop_prob: float = 0.7,     # foreground-biased crop probability (if mask exists)

        # --- splitting options ---
        train_val_test: Tuple[int,int,int] = (70, 15, 15),
        split_seed: str = "v1",
        group_by_case: bool = True,
        verbose_split: bool = True,
    ):
        super().__init__()
        assert split in {"train", "val", "test"}, f"split must be train/val/test, got {split}"

        # resolve image/label dirs
        img_dir = next((os.path.join(root, d) for d in img_dirnames if os.path.isdir(os.path.join(root, d))), None)
        lbl_dir = next((os.path.join(root, d) for d in lbl_dirnames if os.path.isdir(os.path.join(root, d))), None)
        if img_dir is None:
            raise FileNotFoundError(f"Could not find images directory among {img_dirnames} under {root}")

        # labels required for train/val; optional for test
        if split in {"train", "val"} and (lbl_dir is None or not os.path.isdir(lbl_dir)):
            raise FileNotFoundError(
                f"Expected labels directory among {lbl_dirnames} under {root} for split='{split}', but none found."
            )
        has_labels = lbl_dir is not None and os.path.isdir(lbl_dir)

        # ----- robust pairing by Case_<id>_Week<k> -----
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No NIfTI images found under {img_dir}")

        lbl_paths = sorted(glob.glob(os.path.join(lbl_dir, "*.nii*"))) if has_labels else []

        images_by_key: Dict[str, str] = {}
        for p in img_paths:
            k = _key_from_name(os.path.basename(p))
            if not k:
                continue
            prev = images_by_key.get(k)
            # Prefer files that DO NOT look like semantic label when multiple image candidates exist
            if (prev is None) or ("SEMANTIC" in os.path.basename(prev) and "SEMANTIC" not in os.path.basename(p)):
                images_by_key[k] = p

        labels_by_key: Dict[str, str] = {}
        if lbl_paths:
            for p in lbl_paths:
                k = _key_from_name(os.path.basename(p))
                if not k:
                    continue
                prev = labels_by_key.get(k)
                # Prefer files that DO look like labels (contain 'SEMANTIC') if multiples exist
                if (prev is None) or ("SEMANTIC" in os.path.basename(p) and "SEMANTIC" not in os.path.basename(prev)):
                    labels_by_key[k] = p

        pairs_all: List[Tuple[str, Optional[str], str]] = []  # (img, lbl, key)
        for k, ipath in images_by_key.items():
            lpath = labels_by_key.get(k) if has_labels else None
            pairs_all.append((ipath, lpath, k))

        # Fail if labels are expected but not found (train/val)
        if has_labels:
            missing = [os.path.basename(i) for i, l, _ in pairs_all if l is None]
            if split != "test" and missing:
                examples = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"Could not match labels for {len(missing)} case(s) using key 'Case_<id>_Week<k>'. "
                    f"Examples: {examples}"
                )

        # ----- deterministic 70:15:15 split -----
        # Gather groups (either case-root groups or individual keys)
        if group_by_case:
            groups: Dict[str, List[Tuple[str, Optional[str], str]]] = {}
            for ip, lp, k in pairs_all:
                root_id = _case_root_from_key(k) or k
                groups.setdefault(root_id, []).append((ip, lp, k))
            group_keys = sorted(groups.keys())
        else:
            groups = {k: [(ip, lp, k)] for ip, lp, k in pairs_all}
            group_keys = sorted(groups.keys())

        # Deterministic shuffle
        rng = random.Random(split_seed)
        rng.shuffle(group_keys)

        n = len(group_keys)
        t, v, te = train_val_test
        total = float(t + v + te)
        n_train = int(round(n * (t / total)))
        n_val   = int(round(n * (v / total)))
        # ensure sum exactly n
        n_test  = max(0, n - n_train - n_val)

        train_keys = group_keys[:n_train]
        val_keys   = group_keys[n_train:n_train+n_val]
        test_keys  = group_keys[n_train+n_val:]

        select = {"train": train_keys, "val": val_keys, "test": test_keys}[split]
        pairs: List[Tuple[str, Optional[str]]] = []
        for gk in select:
            pairs.extend([(ip, lp) for (ip, lp, _k) in groups[gk]])

        if verbose_split:
            print(f"[Split] groups total={n}  -> train={len(train_keys)}, val={len(val_keys)}, test={len(test_keys)}")
            print(f"[Split] {split}: {len(pairs)} files from {len(select)} group(s)")

        # save config
        self.items = pairs
        self.split = split
        self.target_spacing = target_spacing
        self.crop_to_foreground = crop_to_foreground
        self.patch_size = patch_size
        self.samples_per_vol = samples_per_vol
        self.augment = augment and (split == "train")
        self.for_eval_full_volume = for_eval_full_volume
        self.fg_crop_prob = fg_crop_prob

    def __len__(self):
        if self.patch_size is None or self.for_eval_full_volume:
            return len(self.items)
        return len(self.items) * self.samples_per_vol

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # map idx to actual case (when doing multiple samples per volume)
        true_idx = idx if (self.patch_size is None or self.for_eval_full_volume) else idx // self.samples_per_vol
        img_path, lbl_path = self.items[true_idx]

        # --- load (RAS, aligned spacing)
        img, sp_img = load_nii(img_path)
        if lbl_path is not None:
            lbl, sp_lbl = load_nii(lbl_path)
        else:
            # test split without labels: use empty mask
            lbl, sp_lbl = np.zeros_like(img, dtype=np.float32), sp_img

        # --- resample (image: linear, label: nearest)
        if self.target_spacing is not None:
            img = resample_to_spacing(img, sp_img, self.target_spacing, order=1)
            lbl = resample_to_spacing(lbl, sp_lbl, self.target_spacing, order=0)
            spacing_out = self.target_spacing
        else:
            spacing_out = sp_img  # preserve native spacing in meta

        # --- intensity normalization
        img = percentile_clip_zscore(img)

        # --- optional foreground crop
        if self.crop_to_foreground and lbl is not None:
            bbox = compute_bbox(lbl, pad=(8,8,8))
            if bbox is not None:
                img = img[bbox]; lbl = lbl[bbox]

        # --- choose crop size / full-volume eval
        if self.for_eval_full_volume:
            img = pad_to_multiple(img, 16)
            lbl = pad_to_multiple(lbl, 16)
        elif self.patch_size is not None:
            # foreground-biased crop if mask exists
            if lbl.sum() > 0 and random.random() < self.fg_crop_prob:
                bbox = compute_bbox(lbl, pad=(0,0,0))
                if bbox is not None:
                    zslice, yslice, xslice = bbox
                    img_f = img[zslice, yslice, xslice]
                    lbl_f = lbl[zslice, yslice, xslice]
                    img, lbl = random_crop_3d(img_f, lbl_f, self.patch_size)
                else:
                    img, lbl = random_crop_3d(img, lbl, self.patch_size)
            else:
                img, lbl = random_crop_3d(img, lbl, self.patch_size)

        # --- simple augments (geom on both, intensity on image only)
        if self.augment:
            img, lbl = rand_flip3d(img, lbl, p=0.5)
            if random.random() < 0.3:
                # light in-plane rotation around (H,W); keep order=0 for labels
                angle = random.uniform(-7, 7)
                img = ndi.rotate(img, angle, axes=(1,2), reshape=False, order=1, mode='nearest')
                lbl = ndi.rotate(lbl, angle, axes=(1,2), reshape=False, order=0, mode='nearest')
            img = rand_intensity(img, gamma_range=(0.9,1.1), noise_std=0.02)

        # --- dtypes & tensors [C,D,H,W]
        img = img.astype(np.float32, copy=False)
        lbl = lbl.astype(np.int64,  copy=False)

        img_t = torch.from_numpy(img[None, ...])  # [1,D,H,W]
        lbl_t = torch.from_numpy(lbl[None, ...])  # [1,D,H,W]

        return {
            "image": img_t,
            "label": lbl_t,
            "id": os.path.basename(img_path).replace(".nii.gz","").replace(".nii",""),
            "spacing": torch.tensor(spacing_out, dtype=torch.float32)
        }

# -------------------------- convenience API --------------------------

def make_loaders(
    root: str,
    target_spacing: Tuple[float,float,float] = (3.0, 1.0, 1.0),
    patch_size: Tuple[int,int,int] = (128,128,64),
    batch_size: int = 2,
    workers: int = 4
):
    """
    Convenience: build (train_loader, val_loader) for a generic prostate dataset tree.
    Uses split (70:15:15) via Dataset constructor with deterministic seed.
    """
    common = dict(
        target_spacing=target_spacing,
        crop_to_foreground=True,
        patch_size=patch_size,
        samples_per_vol=4,
        augment=True,
        for_eval_full_volume=False,
        train_val_test=(70, 15, 15),
        split_seed="v1",
        group_by_case=True,
        verbose_split=True,
    )
    train_ds = Prostate3DDataset(root=root, split="train", **common)
    val_ds   = Prostate3DDataset(root=root, split="val",
                                 target_spacing=target_spacing,
                                 crop_to_foreground=False,
                                 patch_size=None,
                                 augment=False,
                                 for_eval_full_volume=True,
                                 train_val_test=(70, 15, 15),
                                 split_seed="v1",
                                 group_by_case=True,
                                 verbose_split=True)

    def collate_pad(batch: List[Dict[str, torch.Tensor]]):
        # Items are uniform-sized within each loader (patches; or padded full volumes).
        imgs = torch.stack([b["image"] for b in batch], dim=0)  # [B,1,D,H,W]
        lbls = torch.stack([b["label"] for b in batch], dim=0)  # [B,1,D,H,W]
        ids  = [b["id"] for b in batch]
        spac = torch.stack([b["spacing"] for b in batch], dim=0)
        return {"image": imgs, "label": lbls, "id": ids, "spacing": spac}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, collate_fn=collate_pad)
    val_loader   = DataLoader(val_ds,   batch_size=1,       shuffle=False,
                              num_workers=workers, pin_memory=True, collate_fn=collate_pad)
    return train_loader, val_loader


def make_loaders_for_hipmri(
    root: str,
    target_spacing: Tuple[float,float,float] = (2.0, 2.0, 2.0),
    patch_size: Tuple[int,int,int] = (128,128,64),
    batch_size: int = 2,
    workers: int = 0
):
    """
    HIP-MRI flavored loaders with expected folder names:
      - Images: 'semantic_MRs_anon'
      - Labels: 'semantic_labels_anon'
    Splits 70:15:15 deterministically by case root.
    """
    common = dict(
        img_dirnames=("semantic_MRs_anon", "images", "imagesTr"),
        lbl_dirnames=("semantic_labels_anon", "labels", "labelsTr"),
        target_spacing=target_spacing,
        train_val_test=(70, 15, 15),
        split_seed="v1",
        group_by_case=True,
        verbose_split=True,
    )

    train_ds = Prostate3DDataset(
        root=root, split="train",
        crop_to_foreground=True, patch_size=patch_size, samples_per_vol=4, augment=True,
        for_eval_full_volume=False, **common
    )

    val_ds = Prostate3DDataset(
        root=root, split="val",
        crop_to_foreground=False, patch_size=None, augment=False,
        for_eval_full_volume=True, **common
    )

    def collate_pad(batch: List[Dict[str, torch.Tensor]]):
        imgs = torch.stack([b["image"] for b in batch], dim=0)
        lbls = torch.stack([b["label"] for b in batch], dim=0)
        ids  = [b["id"] for b in batch]
        spac = torch.stack([b["spacing"] for b in batch], dim=0)
        return {"image": imgs, "label": lbls, "id": ids, "spacing": spac}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, collate_fn=collate_pad)
    val_loader   = DataLoader(val_ds,   batch_size=1,       shuffle=False,
                              num_workers=workers, pin_memory=True, collate_fn=collate_pad)
    return train_loader, val_loader

# -------------------------- metrics --------------------------

def dice_per_channel(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Hard Dice on argmax predictions.
    pred_logits: [N, C, D, H, W]
    target:      [N, 1, D, H, W] integer labels (0..C-1)
    Returns Dice for classes 1..C-1 (ignores background idx 0).
    """
    num_classes = pred_logits.shape[1]
    pred = torch.argmax(pred_logits, dim=1, keepdim=True)  # [N,1,D,H,W]
    dices = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum(dim=[1,2,3,4])
        den   = p.sum(dim=[1,2,3,4]) + t.sum(dim=[1,2,3,4])
        d = (2*inter + eps) / (den + eps)
        dices.append(d)
    if len(dices) == 0:
        return torch.tensor(1.0, device=pred_logits.device)
    return torch.stack(dices, dim=1)  # [N, C-1]


def soft_dice_per_channel(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice using probabilities and one-hot target.
    pred_logits: [N, C, D, H, W], target: [N, 1, D, H, W] (ints)
    Returns per-class Dice for classes 1..C-1.
    """
    N, C = pred_logits.shape[:2]
    probs = torch.softmax(pred_logits, dim=1)
    one_hot = torch.zeros_like(pred_logits).scatter_(1, target, 1)  # [N,C,D,H,W]
    dices = []
    for c in range(1, C):
        p = probs[:, c]
        t = one_hot[:, c]
        inter = (p * t).sum(dim=(1,2,3))
        den   = p.sum(dim=(1,2,3)) + t.sum(dim=(1,2,3))
        dices.append((2*inter + eps) / (den + eps))
    if len(dices) == 0:
        return torch.tensor(1.0, device=pred_logits.device)
    return torch.stack(dices, dim=1)

# -------------------------- script entry: show split sizes --------------------------

if __name__ == "__main__":
    # Auto-discover dataset root (Option B)
    try:
        root = find_dataset_root()
    except FileNotFoundError as e:
        print(e)
        raise

    # Flexible folder-name fallbacks + standard HIP-MRI defaults
    common = dict(
        img_dirnames=("semantic_MRs_anon", "images", "imagesTr"),
        lbl_dirnames=("semantic_labels_anon", "labels", "labelsTr"),
        target_spacing=None,
        crop_to_foreground=False,
        patch_size=None,
        augment=False,
        for_eval_full_volume=True,
        train_val_test=(70, 15, 15),
        split_seed="v1",
        group_by_case=True,
        verbose_split=True,
    )

    splits = {}
    for sp in ("train", "val", "test"):
        ds = Prostate3DDataset(root=root, split=sp, **common)
        splits[sp] = ds

    n_train = len(splits["train"])
    n_val   = len(splits["val"])
    n_test  = len(splits["test"])
    print("\n==== Dataset file counts ====")
    print(f"Train files: {n_train}")
    print(f"Val files:   {n_val}")
    print(f"Test files:  {n_test}")
    print(f"Total files: {n_train + n_val + n_test}")
