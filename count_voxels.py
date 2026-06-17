import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

path = sys.argv[1]
img = nib.load(path)
data = np.asarray(img.dataobj)
mask = data > 0

print(f"shape      : {data.shape}")
print(f"voxel size : {img.header.get_zooms()}")
print(f"dtype      : {data.dtype}")
print(f"min/max    : {data.min()} / {data.max()}")
print(f"unique     : {np.unique(data)}")
print(f"lesion vox : {mask.sum()}  ({100*mask.mean():.2f}%)")

if mask.sum() > 0:
    coords = np.argwhere(mask)
    print(f"lesion bbox: {coords.min(axis=0)} → {coords.max(axis=0)}")
    center = coords.mean(axis=0).astype(int)
else:
    center = np.array(data.shape) // 2

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (axis, idx, title) in zip(axes, [
    (2, center[2], f"axial z={center[2]}"),
    (1, center[1], f"coronal y={center[1]}"),
    (0, center[0], f"sagittal x={center[0]}"),
]):
    sl = np.take(data, idx, axis=axis)
    ax.imshow(sl.T, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(path)
plt.tight_layout()
plt.savefig("lesion_vis.png", dpi=150)
print("saved lesion_vis.png")