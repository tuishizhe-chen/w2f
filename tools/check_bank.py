import torch
for p in ('/root/w2f/checkpoints/edge_bank_128_clean.pt',
          '/root/w2f/checkpoints/edge_bank_128_clean_v3.pt',
          '/root/w2f/checkpoints/edge_bank_128_dilated.pt',
          '/root/w2f/checkpoints/edge_bank_128_dense_dilated.pt'):
    b = torch.load(p, weights_only=True)
    print(p.split('/')[-1], 'ink', round(b.float().mean().item() / 255, 4))
