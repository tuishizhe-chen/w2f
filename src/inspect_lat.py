import torch
l = torch.load('./checkpoints/face_lat.pt', weights_only=True)
print('shape', tuple(l.shape))
print('mean', l.mean().item(), 'std', l.std().item())
print('min', l.min().item(), 'max', l.max().item())
q = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])
sub = l.abs().flatten()[torch.randperm(l.numel())[:1000000]]
print('|x| quantiles 0/1/50/99/100 =', sub.quantile(q).tolist())
# per-channel std
chs = l.std(dim=(0, 2, 3))
print('per-channel std min/max/mean:', chs.min().item(), chs.max().item(), chs.mean().item())
# pairwise distance distribution between random reals
import torch
N = l.shape[0]
sel = torch.randperm(N)[:256]
flat = l[sel].flatten(1)
d = torch.cdist(flat, flat)
mask = torch.eye(256, dtype=torch.bool); d = d.masked_fill(mask, 0)
print('pairwise dist among 256 reals: mean', d.sum().item()/(256*255), 'min', d[~mask].min().item(), 'max', d.max().item())
