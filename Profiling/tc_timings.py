import torch
import time

from torch.nn.functional import conv3d

torch_3d = conv3d
from torch.nn import Conv3d
from Group3DConvTC.tc_conv import Grouped3D


######### Params
B = 4
G = 64
O = 64
C = 64
K = 3


########## Init Weights and Layers
X = torch.randn(B, C, 50, 50, 50, requires_grad=True).cuda()
X1 = X.clone().data
X1.requires_grad = True

tc_gc = Grouped3D(C, O, K, G, from_cache=True, cache_file='/Grouped3DConvPyTorch/grp-b4-hw50-c64-o64-g64-k3.pt').cuda()
W = tc_gc.W.clone().data.view(O, C / G, K, K, K).cuda()
W = torch.nn.Parameter(W)

torch_3d = Conv3d(C, O, K, groups=G, bias=False)
torch_3d.weight = W


########## Timings
# Init cuda
torch_out = torch_3d(X1)
torch_out.sum().backward()
torch.cuda.synchronize()
torch_out = torch_3d(X1)
torch_out.sum().backward()
torch.cuda.synchronize()

# Torch
t = time.time()
for i in range(100):
    torch_out = torch_3d(X1)
    torch_out.sum().backward()
    torch_3d.zero_grad()
print('Torch: ', time.time() - t, 's')


tc_out = tc_gc(X)
tc_out.sum().backward()
torch.cuda.synchronize()
tc_out = tc_gc(X)
tc_out.sum().backward()
torch.cuda.synchronize()

# TC
t = time.time()
for i in range(100):
    tc_out = tc_gc(X)
    tc_out.sum().backward()
    tc_gc.zero_grad()
print('TC: ', time.time() - t, 's')

# Make sure weight grads are same
print('Max difference in gradients: ', torch.max(tc_gc.W.grad.view(W.shape) - W.grad))

