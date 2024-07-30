import numpy as np
import torch
from torch.ao.quantization.fake_quantize import FakeQuantize

# means = torch.rand((1000, 3)) * 20
# means = torch.nn.Parameter(means)
# means_mins = torch.zeros(3)
# means_maxs = torch.ones(3) * 20
# # means_mins = torch.amin(means, dim=0).detach()
# # means_maxs = torch.amax(means, dim=0).detach()
# print(means_mins, means_maxs)

# scales = (means_maxs - means_mins) / 255.0
# zero_points = torch.zeros((3,))
# x = torch.fake_quantize_per_channel_affine(means, scales, zero_points, axis=1, quant_min=0, quant_max=255)
# diff = means - x
# print(diff.abs().max(), diff.mean())

# means_norm = (means - means_mins) / (means_maxs - means_mins)
# means_norm = means_norm.detach().cpu().numpy()
# means_q = (means_norm * 256 - 0.5).round().astype(np.uint8)
# # means_q = (means_norm * 1023).astype(np.uint16)

# means_q = torch.tensor(means_q, dtype=torch.float32)
# means_norm = (means_q + 0.5) / 256.0
# # means_norm = means_q / 1023.0
# x = means_norm * (means_maxs - means_mins) + means_mins
# diff = means - x
# print(diff.abs().max(), diff.mean())

import torch
from torchpq.clustering import KMeans


def check_equal(sorted_indices):
    x = np.load("x.npz")
    arr0 = x["arr"][sorted_indices]

    arr1 = np.load("y0.npz")["arr1"]
    assert np.allclose(arr0, arr1)

    y1 = np.load("y1.npz")
    labels = y1["labels"]
    deltas = y1["deltas"].transpose(1, 0)
    centroids = y1["centroids"].transpose(1, 0)

    arr2 = centroids[labels] + deltas
    print(np.abs(arr0 - arr2).max())
    assert np.allclose(arr0, arr2)


ckpt_path = "examples/results/360_v2/3dgs/garden/ckpts/ckpt_29999.pt"
splats = torch.load(ckpt_path)["splats"]
n_gs = len(splats["means"])
params = [
    splats[k].reshape(n_gs, -1)
    for k in ["quats", "means", "opacities", "quats", "scales", "sh0"]
]
x = torch.cat(params, dim=-1).cuda()

mins = torch.amin(x, dim=0)
maxs = torch.amax(x, dim=0)
x = (x - mins) / (maxs - mins)
x = (x * (2**8 - 1)).round()

arr = x.detach().cpu().numpy()
arr = arr.astype(np.uint8)
np.savez_compressed("x.npz", arr=arr)

kmeans = KMeans(n_clusters=2**16, distance="euclidean", verbose=True)
labels = kmeans.fit(x.permute(1, 0).contiguous())
centroids = kmeans.centroids.permute(1, 0).round()
sorted_indices = torch.argsort(labels)

arr1 = x[sorted_indices]
arr1 = arr1.detach().cpu().numpy()
arr1 = arr1.astype(np.uint8)
np.savez_compressed("y0.npz", arr1=arr1)


deltas = x - centroids[labels]
labels = labels[sorted_indices]
deltas = deltas[sorted_indices]

# test0 = centroids[labels] + deltas
# print((test0 - x[sorted_indices]).abs().max())


labels = labels.detach().cpu().numpy()
centroids = centroids.detach().cpu().numpy().transpose(1, 0)
deltas = deltas.detach().cpu().numpy().transpose(1, 0)
print(labels.shape, labels.min(), labels.max())
print(centroids.shape, centroids.min(), centroids.max())
print(deltas.shape, deltas.min(), deltas.max())

meta = {
    "labels": np.ascontiguousarray(labels.astype(np.uint16)),
    "centroids": np.ascontiguousarray(centroids.astype(np.uint8)),
    "deltas": np.ascontiguousarray(deltas.astype(np.int8)),
}
# np.savez_compressed("y_labels.npz", labels=meta["labels"])
# np.savez_compressed("y_centroids.npz", centroids=meta["centroids"])
# np.savez_compressed("y_deltas.npz", deltas=meta["deltas"])
np.savez_compressed("y1.npz", **meta)

sorted_indices = sorted_indices.cpu().numpy()
check_equal(sorted_indices)


# sorted_indices = np.argsort(x[:,0])
# y = x[sorted_indices, :]
# print(y.shape)
# np.savez_compressed("y.npz", arr=y)
