import os, torch
print('--- ckpts in dir ---')
for f in sorted(os.listdir('/root/w2f/checkpoints')):
    if 'letter' in f.lower():
        print(f, os.path.getsize(f'/root/w2f/checkpoints/{f}'))
print('--- letter_bank.pt detail ---')
b = torch.load('/root/w2f/checkpoints/letter_bank.pt', weights_only=True)
print('shape:', tuple(b.shape), 'dtype:', b.dtype, 'min/max:', b.min().item(), b.max().item())
