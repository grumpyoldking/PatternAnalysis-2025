import nibabel as nib, numpy as np, os
label_path = r"...\semantic_labels_anon\YOUR_CASE.nii.gz"  # pick a known match
lab = nib.load(label_path).get_fdata()
print("Shape:", lab.shape, "dtype:", lab.dtype, "min/max:", lab.min(), lab.max())
u, c = np.unique(lab, return_counts=True)
print("Unique values:", dict(zip(u.astype(int), c)))
