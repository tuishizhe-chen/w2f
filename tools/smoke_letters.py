import sys
sys.path.insert(0, '/root/w2f/src')
import face_letters_pixel as fl
print('imports OK')
print('LetterImgGen ctor:', list(fl.LetterImgGen.__init__.__code__.co_varnames[:8]))
import inspect
sig = inspect.signature(fl.train)
print('train fn:', list(sig.parameters))
import torch
# also try to instantiate
G = fl.LetterImgGen(d_noise=128, n_classes=26, letter_size=64, base=64,
                    head_refine=2, sigmoid_t=1.0)
print('G params:', sum(p.numel() for p in G.parameters()) / 1e6, 'M')
# forward smoke
eps = torch.randn(2, 128)
labels = torch.randint(0, 26, (2, 12))
img, theta = G(eps, labels)
print('img shape:', tuple(img.shape), 'theta shape:', tuple(theta.shape))
print('theta r range:', theta[..., 0].min().item(), theta[..., 0].max().item())
# STN test
flat = img.view(2 * 12, 1, 64, 64)
inv_s = 1.0 / theta[..., 0].reshape(-1)
tx = theta[..., 1].reshape(-1)
ty = theta[..., 2].reshape(-1)
placed = fl.stn_place(flat, inv_s, tx, ty, 128)
print('placed shape:', tuple(placed.shape))
canvas = placed.view(2, 12, 1, 128, 128).max(dim=1).values
print('canvas shape:', tuple(canvas.shape), 'mean:', canvas.mean().item())
print('all OK')
