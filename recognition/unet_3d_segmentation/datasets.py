# dataloaders/prostate3d.py
# deps: pip/conda install torch nibabel scipy numpy
import os, glob, random
from typing import Tuple, Optional, List, Dict
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage as ndi


# -------------------------- helpers --------------------------

def load_nii(path: str):
    """
    Load a NIfTI and reorient to canonical RAS so that header zooms align with axes.
    Returns:
      data  (np.float32): array shaped (D, H, W) == (Z, Y, X)
      space (tuple): voxel spacing in mm as (Dz, Dy, Dx) aligned to (D, H, W)
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
    # spacing larger than target -> zoom > 1 (upsample)
    return ndi.zoom(img, zoom=zoom, order=order)


def percentile_clip_zscore(x: np.ndarray, pmin=0.5, pmax=99.5, eps=1e-6) -> np.ndarray:
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
    """Pad (D,H,W) array with zeros so each dim is a multiple of m (pad at the end only)."""
    D, H, W = arr.shape
    pD = (m - D % m) % m
    pH = (m - H % m) % m
    pW = (m - W % m) % m
    return np.pad(arr, ((0, pD), (0, pH), (0, pW)), mode="constant")


def rand_flip3d(img, msk, p=0.5):
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


# -------------------------- Dataset --------------------------

class Prostate3DDataset(Dataset):
    """
    Directory layout:
      root/
        images/ or imagesTr/
          case_000.nii.gz ...
        labels/ or labelsTr/
          case_000.nii.gz ...
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
        fg_crop_prob: float = 0.7      # foreground-biased crop probability (if mask exists)
    ):
        super().__init__()
        # resolve image/label dirs
        img_dir = next((os.path.join(root, d) for d in img_dirnames if os.path.isdir(os.path.join(root, d))), None)
        lbl_dir = next((os.path.join(root, d) for d in lbl_dirnames if os.path.isdir(os.path.join(root, d))), None)
        if img_dir is None:
            raise FileNotFoundError(f"Could not find images directory among {img_dirnames} under {root}")
        has_labels = lbl_dir is not None and os.path.isdir(lbl_dir)

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No NIfTI images found under {img_dir}")

        id2img = {os.path.splitext(os.path.basename(p))[0].replace(".nii",""): p for p in img_paths}

        pairs = []
        for cid, ipath in id2img.items():
            if has_labels:
                candidates = [
                    os.path.join(lbl_dir, os.path.basename(ipath)),
                    os.path.join(lbl_dir, cid + ".nii.gz"),
                    os.path.join(lbl_dir, cid + ".nii"),
                ]
                lpath = next((c for c in candidates if os.path.exists(c)), None)
            else:
                lpath = None
            pairs.append((ipath, lpath))

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
            lbl, sp_lbl = np.zeros_like(img, dtype=np.float32), sp_img

        # --- resample (image: linear, label: nearest), using their own spacings
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
    train_ds = Prostate3DDataset(
        root=root,
        split="train",
        target_spacing=target_spacing,
        crop_to_foreground=True,
        patch_size=patch_size,
        samples_per_vol=4,
        augment=True,
        for_eval_full_volume=False,
        fg_crop_prob=0.7
    )
    val_ds = Prostate3DDataset(
        root=root,
        split="val",
        target_spacing=target_spacing,
        crop_to_foreground=False,
        patch_size=None,                 # evaluate on full volume (padded)
        augment=False,
        for_eval_full_volume=True
    )

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

# at bottom of datasets.py
from typing import Tuple, List, Dict
import torch
from torch.utils.data import DataLoader

# import the class from above in the same file:
# from .prostate3d import Prostate3DDataset  # if you split files
# Here we assume Prostate3DDataset is in this file.

def make_loaders_for_hipmri(
    root: str,
    target_spacing: Tuple[float,float,float] = (2.0, 2.0, 2.0),
    patch_size: Tuple[int,int,int] = (128,128,64),
    batch_size: int = 2,
    workers: int = 0
):
    train_ds = Prostate3DDataset(
        root=root,
        split="train",
        img_dirnames=("semantic_MRs_anon",),       # <-- your folders
        lbl_dirnames=("semantic_labels_anon",),    # <-- your folders
        target_spacing=target_spacing,
        crop_to_foreground=True,
        patch_size=patch_size,
        samples_per_vol=4,
        augment=True,
        for_eval_full_volume=False,
        fg_crop_prob=0.7
    )

    val_ds = Prostate3DDataset(
        root=root,
        split="val",
        img_dirnames=("semantic_MRs_anon",),
        lbl_dirnames=("semantic_labels_anon",),
        target_spacing=target_spacing,
        crop_to_foreground=False,
        patch_size=None,                 # full volume (padded to /16)
        augment=False,
        for_eval_full_volume=True
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

