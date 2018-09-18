import torch
from torch.nn import Conv2d, Conv3d
import time

######### Params
B = 4
G = 64
O = 64
C = 64
K = 3

######### Init
X2d = torch.randn(B, C, 50, 50, requires_grad=True).cuda()
X3d = torch.randn(B, C, 50, 50, 50, requires_grad=True).cuda()

c2d_g1 = Conv2d(C, O, K, groups=1).cuda()
c2d_gc = Conv2d(C, O, K, groups=C).cuda()

c3d_g1 = Conv3d(C, O, K, groups=1).cuda()
c3d_gc = Conv3d(C, O, K, groups=C).cuda()


######### Timings

torch_out = c2d_g1(X2d)
torch_out.sum().backward()
torch.cuda.synchronize()
torch_out = c2d_g1(X2d)
torch_out.sum().backward()
torch.cuda.synchronize()

# 2D, 1 Group
t = time.time()
for i in range(1000):
    torch_out = c2d_g1(X2d)
    torch_out.sum().backward()
    c2d_g1.zero_grad()
print('2D, 1 Group: ', time.time() - t)


torch_out = c2d_gc(X2d)
torch_out.sum().backward()
torch.cuda.synchronize()
torch_out = c2d_gc(X2d)
torch_out.sum().backward()
torch.cuda.synchronize()

# 2D, C groups
t = time.time()
for i in range(1000):
    torch_out = c2d_gc(X2d)
    torch_out.sum().backward()
    c2d_gc.zero_grad()
print('2D, C groups: ', time.time() - t)


torch_out = c2d_g1(X2d)
torch_out.sum().backward()
torch.cuda.synchronize()
torch_out = c2d_g1(X2d)
torch_out.sum().backward()
torch.cuda.synchronize()


# 3D, 1 Group
t = time.time()
for i in range(1000):
    torch_out = c3d_g1(X3d)
    torch_out.sum().backward()
    c3d_g1.zero_grad()
print('3D, 1 Group: ', time.time() - t)


torch_out = c3d_gc(X3d)
torch_out.sum().backward()
torch.cuda.synchronize()
torch_out = c3d_gc(X3d)
torch_out.sum().backward()
torch.cuda.synchronize()

# 3D, C groups
t = time.time()
for i in range(1000):
    torch_out = c3d_gc(X3d)
    torch_out.sum().backward()
    c3d_gc.zero_grad()
print('3D, C groups: ', time.time() - t)
