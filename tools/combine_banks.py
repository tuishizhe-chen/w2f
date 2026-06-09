"""Build a variable-thickness bank by union-ing a thin-all-edges bank with
a thick-strong-edges bank. Both must have been generated from the same
sorted image list (which build_edge_bank_64_clean.py does), so they align."""
import sys, torch


def main():
    thin_path, thick_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    bt = torch.load(thin_path, weights_only=True)
    bk = torch.load(thick_path, weights_only=True)
    assert bt.shape == bk.shape, f'shape mismatch {bt.shape} vs {bk.shape}'
    combined = torch.maximum(bt, bk)
    torch.save(combined, out_path)
    bt_ink = bt.float().mean().item() / 255
    bk_ink = bk.float().mean().item() / 255
    cb_ink = combined.float().mean().item() / 255
    print(f'thin ({thin_path.split("/")[-1]}): ink={bt_ink:.4f}')
    print(f'thick ({thick_path.split("/")[-1]}): ink={bk_ink:.4f}')
    print(f'varied (union): ink={cb_ink:.4f}  → saved {out_path}')


if __name__ == '__main__':
    main()
