
import numpy as np
import torch
from torch.ao.quantization.fake_quantize import FakeQuantize

means = torch.rand((1000, 3)) * 20
means = torch.nn.Parameter(means)
means_mins = torch.zeros(3)
means_maxs = torch.ones(3) * 20
# means_mins = torch.amin(means, dim=0).detach()
# means_maxs = torch.amax(means, dim=0).detach()
print(means_mins, means_maxs)

scales = (means_maxs - means_mins) / 255.0
zero_points = torch.zeros((3,))
x = torch.fake_quantize_per_channel_affine(means, scales, zero_points, axis=1, quant_min=0, quant_max=255)
diff = means - x
print(diff.abs().max(), diff.mean())

means_norm = (means - means_mins) / (means_maxs - means_mins)
means_norm = means_norm.detach().cpu().numpy()
means_q = (means_norm * 256 - 0.5).round().astype(np.uint8)
# means_q = (means_norm * 1023).astype(np.uint16)

means_q = torch.tensor(means_q, dtype=torch.float32)
means_norm = (means_q + 0.5) / 256.0
# means_norm = means_q / 1023.0
x = means_norm * (means_maxs - means_mins) + means_mins
diff = means - x
print(diff.abs().max(), diff.mean())

