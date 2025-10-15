# dataloaders/prostate3d.py
# pip install nibabel scipy numpy torch
import os, glob, math, random
from typing import Tuple, Optional, List, Dict
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage as ndi


# -------------------------- helpers --------------------------

def load_nii(path: str):
    nimg = nib.load(path)
    data = nimg.get_fdata(dtype=np.float32)  # (Z,Y,X) in nib, or (H,W,D) depending on file
    # Standardize to (D, H, W) with D as axial slices (z)
    # NIfTI is typically (X,Y,Z). We’ll reorder to (Z,Y,X) => (D,H,W)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got shape {data.shape}")
    data = np.transpose(data, (2, 1, 0))  # (Z,Y,X)
    spacing = nimg.header.get_zooms()[:3]  # (X,Y,Z) mm/voxel
    spacing = spacing[::-1]                # -> (Z,Y,X) to match our order
    return data, spacing  # np.float32, tuple(float,float,float)


def resample_to_spacing(
    img: np.ndarray,
    spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int
) -> np.ndarray:
    """Resample 3D (D,H,W) to target_spacing. order: 1=linear, 0=nearest."""
    zoom = tuple(s / t for s, t in zip(spacing, target_spacing))  # factor per axis
    # When spacing is larger than target_spacing, zoom>1 => upsample
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
    """Random crop (D,H,W) to size (d,h,w). Pads if needed."""
    d, h, w = img.shape
    dz, dy, dx = size
    pad_z = max(0, dz - d); pad_y = max(0, dy - h); pad_x = max(0, dx - w)
    if pad_z or pad_y or pad_x:
        # pad equally on both sides
        pz0, pz1 = pad_z // 2, pad_z - pad_z//2
        py0, py1 = pad_y // 2, pad_y - pad_y//2
        px0, px1 = pad_x // 2, pad_x - pad_x//2
        img = np.pad(img, ((pz0,pz1),(py0,py1),(px0,px1)), mode='constant')
        msk = np.pad(msk, ((pz0,pz1),(py0,py1),(px0,px1)), mode='constant')
        d, h, w = img.shape
    z0 = random.randint(0, d - dz) if d > dz else 0
    y0 = random.randint(0, h - dy) if h > dy else 0
    x0 = random.randint(0, w - dx) if w > dx else 0
    return img[z0:z0+dz, y0:y0+dy, x0:x0+dx], msk[z0:z0+dz, y0:y0+dy, x0:x0+dx]


def center_crop_or_pad(img, msk, size: Tuple[int,int,int]):
    d, h, w = img.shape
    dz, dy, dx = size
    # pad then center crop
    pad_z = max(0, dz - d); pad_y = max(0, dy - h); pad_x = max(0, dx - w)
    if pad_z or pad_y or pad_x:
        pz0, pz1 = pad_z // 2, pad_z - pad_z//2
        py0, py1 = pad_y // 2, pad_y - pad_y//2
        px0, px1 = pad_x // 2, pad_x - pad_x//2
        img = np.pad(img, ((pz0,pz1),(py0,py1),(px0,px1)), mode='constant')
        msk = np.pad(msk, ((pz0,pz1),(py0,py1),(px0,px1)), mode='constant')
    d, h, w = img.shape
    z0 = max(0, (d - dz)//2); y0 = max(0, (h - dy)//2); x0 = max(0, (w - dx)//2)
    return img[z0:z0+dz, y0:y0+dy, x0:x0+dx], msk[z0:z0+dz, y0:y0+dy, x0:x0+dx]


def rand_flip3d(img, msk, p=0.5):
    if random.random() < p:
        img = img[::-1, ...]; msk = msk[::-1, ...]
    if random.random() < p:
        img = img[:, ::-1, :]; msk = msk[:, ::-1, :]
    if random.random() < p:
        img = img[:, :, ::-1]; msk = msk[:, :, ::-1]
    return img, msk


def rand_intensity(img, gamma_range=(0.9, 1.1), noise_std=0.03):
    g = random.uniform(*gamma_range)
    # shift to positive, apply gamma-like contrast, then re-center
    x = img
    mn = x.min()
    x = x - mn + 1e-6
    x = x ** g
    x = x + mn
    if noise_std > 0:
        x = x + np.random.normal(0, noise_std, size=x.shape).astype(np.float32)
    return x


# -------------------------- Dataset --------------------------

class Prostate3DDataset(Dataset):
    """
    Expects directory layout:
      root/
        images/  (or imagesTr/)
          case_000.nii.gz ...
        labels/  (or labelsTr/)
          case_000.nii.gz ...
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dirnames: Tuple[str,str] = ("images", "imagesTr"),
        lbl_dirnames: Tuple[str,str] = ("labels", "labelsTr"),
        target_spacing: Optional[Tuple[float,float,float]] = None,  # e.g. (3.0, 1.0, 1.0) (D,H,W) mm
        crop_to_foreground: bool = True,
        patch_size: Optional[Tuple[int,int,int]] = (128,128,64),
        samples_per_vol: int = 1,       # used when patch_size is not None
        augment: bool = True,
        for_eval_full_volume: bool = False
    ):
        super().__init__()
        # find dirs
        img_dir = None
        lbl_dir = None
        for d in img_dirnames:
            p = os.path.join(root, d)
            if os.path.isdir(p):
                img_dir = p; break
        for d in lbl_dirnames:
            p = os.path.join(root, d)
            if os.path.isdir(p):
                lbl_dir = p; break
        if img_dir is None:
            raise FileNotFoundError(f"Could not find images directory in {img_dirnames} under {root}")
        # labels might be absent for test set; handle gracefully
        has_labels = lbl_dir is not None and os.path.isdir(lbl_dir)

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No NIfTI images found under {img_dir}")
        id2img = {os.path.splitext(os.path.basename(p))[0].replace(".nii",""): p for p in img_paths}

        pairs = []
        for cid, ipath in id2img.items():
            if has_labels:
                # try exact filename match first
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

    def __len__(self):
        if self.patch_size is None or self.for_eval_full_volume:
            return len(self.items)
        return len(self.items) * self.samples_per_vol

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # map idx to case
        true_idx = idx if (self.patch_size is None or self.for_eval_full_volume) else idx // self.samples_per_vol
        img_path, lbl_path = self.items[true_idx]

        # load
        img, sp_img = load_nii(img_path)
        if lbl_path is not None:
            lbl, sp_lbl = load_nii(lbl_path)
            # quick consistency check
            if img.shape != lbl.shape:
                # if shapes differ (rare), resample label to img spacing/shape as nearest
                # but we will resample both to target spacing soon anyway.
                pass
        else:
            lbl = np.zeros_like(img, dtype=np.float32)

        # resample if requested
        if self.target_spacing is not None:
            img = resample_to_spacing(img, sp_img, self.target_spacing, order=1)
            lbl = resample_to_spacing(lbl, sp_img, self.target_spacing, order=0)

        # intensity norm
        img = percentile_clip_zscore(img)

        # optional foreground crop
        if self.crop_to_foreground:
            bbox = compute_bbox(lbl, pad=(8,8,8))
            if bbox is not None:
                img = img[bbox]; lbl = lbl[bbox]

        # choose crop size
        if self.for_eval_full_volume:
            # pad to multiples of 16 for U-Net down/upsampling convenience
            def pad_to_mult(x, m=16):
                pad = []
                for s in x.shape[::-1]:  # W,H,D
                    r = (-s) % m
                    pad.extend([0, r])
                pad = tuple(pad)  # (leftW,rightW,leftH,rightH,leftD,rightD)
                return np.pad(x, ((0, pad[-1]), (0, pad[-3]), (0, pad[-5])), mode='constant')
            img = pad_to_mult(img)
            lbl = pad_to_mult(lbl)
        elif self.patch_size is not None:
            # foreground-biased crop 50% of the time if mask exists
            if lbl.sum() > 0 and random.random() < 0.5:
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
        else:
            # fixed center crop or pad to a reasonable size if given
            pass

        # augments
        if self.augment:
            img, lbl = rand_flip3d(img, lbl, p=0.5)
            # light 3D rotation (nearest on mask, linear on image)
            if random.random() < 0.3:
                angles = [random.uniform(-7, 7) for _ in range(3)]  # degrees
                img = ndi.rotate(img, angles[0], axes=(1,2), reshape=False, order=1, mode='nearest')
                lbl = ndi.rotate(lbl, angles[0], axes=(1,2), reshape=False, order=0, mode='nearest')
            img = rand_intensity(img, gamma_range=(0.9,1.1), noise_std=0.02)

        # to tensors [C,D,H,W]
        img_t = torch.from_numpy(img[None, ...].astype(np.float32))
        lbl_t = torch.from_numpy(lbl[None, ...].astype(np.int64))  # keep labels as long for CE/Dice

        return {
            "image": img_t,          # shape [1, D, H, W]
            "label": lbl_t,          # shape [1, D, H, W]
            "id": os.path.basename(img_path).replace(".nii.gz","").replace(".nii",""),
            "spacing": torch.tensor(self.target_spacing if self.target_spacing else (0,0,0), dtype=torch.float32)
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
        augment=True
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
        # All items are already same size if patches; for full volumes we ensured padding.
        imgs = torch.stack([b["image"] for b in batch], dim=0)
        lbls = torch.stack([b["label"] for b in batch], dim=0)
        ids  = [b["id"] for b in batch]
        spac = torch.stack([b["spacing"] for b in batch], dim=0)
        return {"image": imgs, "label": lbls, "id": ids, "spacing": spac}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=collate_pad)
    val_loader   = DataLoader(val_ds,   batch_size=1,       shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate_pad)
    return train_loader, val_loader


# -------------------------- metrics --------------------------

def dice_per_channel(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_logits: [N, C, D, H, W] raw scores
    target:      [N, 1, D, H, W] integer labels (0..C-1)
    returns Dice for classes 1..C-1 (ignores background idx 0)
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
        return torch.tensor(1.0)  # trivial if only background
    return torch.stack(dices, dim=1)  # [N, C-1]
