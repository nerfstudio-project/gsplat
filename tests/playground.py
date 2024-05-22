import torch
import torch.nn.functional as F

torch.manual_seed(42)

a = torch.randn(4)
a = F.normalize(a, p=2, dim=-1)
a.requires_grad = True

v = torch.randn(4)

norm = torch.norm(a, p=2, dim=-1, keepdim=True)
b = a / norm 
(b * v).sum().backward()
print (a.grad)

# derivative of 1/norm(a) w.r.t. a
# norm(a) = sqrt(a1^2 + a2^2 + a3^2 + a4^2)

# vbb = b * b * v

# grad_n = (
#     (v[0] 
#      - v[0] * b[0] * b[0] 
#      - v[1] * b[1] * b[0] 
#      - v[2] * b[2] * b[0] 
#      - v[3] * b[3] * b[0]) / norm,
#     #
#     (1 - b[1] * b[1]) * v[1] / norm - 
#     (v[0] * b[0] + v[2] * b[2] + v[3] * b[3]) * b[1] / norm,   
# )

grad_n = (v - v @ b * b) / norm
print (grad_n, v)
print (a, b)