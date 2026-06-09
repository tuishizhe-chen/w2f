import sys, torch
sys.path.insert(0, '/root/w2f/src')
import face_letters_v2 as L
print('imports OK')
G = L.LetterPlacer(d_noise=128, n_classes=26, K=12).cuda()
print('G params:', sum(p.numel() for p in G.parameters()) / 1e6, 'M')
eps = torch.randn(4, 128, device='cuda')
labels = torch.randint(0, 26, (4, 12), device='cuda')
theta = G(eps, labels)
print('theta shape:', tuple(theta.shape))
print('r range:', theta[..., 0].min().item(), theta[..., 0].max().item())
print('tx range:', theta[..., 1].min().item(), theta[..., 1].max().item())
# bank load + STN test
bank = torch.load('/root/w2f/checkpoints/letter_bank.pt', weights_only=True).float() / 255.
bank = bank.cuda()
flat_lab = labels.reshape(-1)
idx = torch.randint(0, 500, (4, 12), device='cuda').reshape(-1)
imgs = bank[flat_lab, idx].unsqueeze(1)
print('imgs shape:', tuple(imgs.shape))
r = theta[..., 0].reshape(-1)
tx = theta[..., 1].reshape(-1)
ty = theta[..., 2].reshape(-1)
placed = L.stn_place(imgs, 1.0/r, tx, ty, 128)
canvas = placed.view(4, 12, 1, 128, 128).max(dim=1).values
print('canvas:', tuple(canvas.shape), 'mean:', canvas.mean().item())
print('all OK')
