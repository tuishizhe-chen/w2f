"""Pretrain a frozen LetterCNN on the aug letter bank, for use as a
deformation-invariant letter-IDENTITY loss in W2F training.

It does NOT replace D1 drift (hard requirement) — it's an additional signal:
D1 drift keeps patches on the real-letter manifold + repels same-class peers,
while the CNN asserts "this patch is recognizable as letter c" in a way that
tolerates the deformation faces need (a Z stretched along a jaw still reads Z).

Usage:
  python .local/train_letter_cnn.py <aug_bank.pt> <out.pt> [steps]
e.g.
  python .local/train_letter_cnn.py checkpoints/aug_letter_bank_32_mild_50000.pt \
         checkpoints/letter_cnn_32_mild.pt 2000
"""
import sys
import torch

sys.path.insert(0, '/root/w2f/src')
from classifier import train_classifier, save

bank_path = sys.argv[1]
out_path = sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bank = torch.load(bank_path, weights_only=True)   # [26, N, 1, H, W]
print(f"[train_cnn] loaded bank {tuple(bank.shape)} from {bank_path}", flush=True)
# squeeze channel dim -> [26, N, H, W] as train_classifier expects
if bank.ndim == 5:
    bank = bank[:, :, 0]
assert bank.ndim == 4 and bank.shape[0] == 26, f"bad bank shape {tuple(bank.shape)}"
H = bank.shape[-1]
assert H == 32, f"classifier is 32x32; bank is {H}x{H}. Use a 32px bank or adapt."

# build_letter_bank.py saves uint8 [0,255]; train_classifier wants float [0,1].
if bank.dtype == torch.uint8:
    bank = bank.float() / 255.0

model = train_classifier(bank, device, steps=steps, batch=256, lr=1e-3)
save(model, out_path)
print(f"[train_cnn] saved -> {out_path}", flush=True)
