"""Sample faces from each bank and stitch them into per-bank grid PNGs."""
import sys, os, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    banks = sys.argv[1].split(',')
    out_dir = sys.argv[2]
    n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(42)
    # use SAME indices across banks so visual comparison is meaningful
    idx_master = None
    for p in banks:
        b = torch.load(p, weights_only=True)
        if idx_master is None:
            idx_master = torch.randperm(b.shape[0])[:n_show]
        ink = b.float().mean().item() / 255
        rows, cols = 2, n_show // 2
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4),
                                 facecolor='#0d0f14')
        for c in range(cols):
            for r in range(rows):
                ax = axes[r][c]
                ax.imshow(b[idx_master[r * cols + c]].numpy(),
                          cmap='gray', vmin=0, vmax=255)
                ax.axis('off')
        name = p.split('/')[-1].replace('.pt', '')
        fig.suptitle(f'{name}   ink={ink:.3f}',
                     color='white', fontsize=11, y=0.99)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01,
                            hspace=0.05, wspace=0.04)
        out_path = os.path.join(out_dir, f'{name}.png')
        plt.savefig(out_path, dpi=110, facecolor='#0d0f14')
        plt.close()
        print(f'saved {out_path}  ink={ink:.3f}')


if __name__ == '__main__':
    main()
