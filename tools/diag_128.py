"""Diagnose why 128-res gen output stays at mid values.
Loads a trained G, samples eps, computes pre-sigmoid logit distribution
and post-sigmoid pixel value histogram."""
import sys, torch
sys.path.insert(0, '/root/w2f/src')
from face_drift_pixel import PixelGen


def hist_str(vals, bins=10, lo=0.0, hi=1.0):
    h = torch.histc(vals, bins=bins, min=lo, max=hi)
    total = h.sum().item()
    pcts = (h / total * 100).tolist()
    edges = [round(lo + (hi - lo) * i / bins, 2) for i in range(bins + 1)]
    return ' | '.join(f'{edges[i]:.2f}-{edges[i+1]:.2f}: {pcts[i]:.1f}%'
                      for i in range(bins))


def main():
    ckpt_path = sys.argv[1]
    ck = torch.load(ckpt_path, weights_only=False, map_location='cuda')
    args = ck['args']
    print(f'loaded {ckpt_path}')
    print(f'  args: size auto from bank, base={args.get("base")}, '
          f'sharpness={args.get("sharpness")}, sigmoid_t={args.get("sigmoid_t")}')

    # detect size from bank (resolve relative path against /root/w2f)
    import os
    bank_path = args['bank']
    if not os.path.isabs(bank_path):
        bank_path = os.path.join('/root/w2f', bank_path)
    bank = torch.load(bank_path, weights_only=True)
    S = bank.shape[-1]
    print(f'  bank size: {S}, real_ink={bank.float().mean().item()/255:.4f}')

    G = PixelGen(d_noise=args.get('d_noise', 128), base=args.get('base', 128),
                 size=S, sigmoid_t=args.get('sigmoid_t', 1.0)).cuda()
    G.load_state_dict(ck.get('G_ema', ck['G']))
    G.eval()

    with torch.no_grad():
        eps = torch.randn(64, args.get('d_noise', 128), device='cuda')
        # we need to capture pre-sigmoid value
        # PixelGen.forward: sigmoid(head(x) * sigmoid_t).
        # easiest: monkey-patch head fwd to also return its raw output.
        B = eps.shape[0]
        x = G.fc(eps).view(B, G.base, 4, 4)
        x = G.up(x)
        raw = G.head(x)              # pre-sigmoid logits
        post_t = raw * G.sigmoid_t   # after sigmoid_t scaling
        out = torch.sigmoid(post_t)  # final output

        print(f'\n[pre-sigmoid logits] mean={raw.mean():.3f} std={raw.std():.3f} '
              f'min={raw.min():.3f} max={raw.max():.3f}')
        print(f'[after sigmoid_t={G.sigmoid_t:.1f}] mean={post_t.mean():.3f} '
              f'std={post_t.std():.3f} min={post_t.min():.3f} max={post_t.max():.3f}')
        print(f'[final output 0-1] mean={out.mean():.4f} std={out.std():.4f} '
              f'min={out.min():.4f} max={out.max():.4f}')

        print(f'\n[output histogram, all pixels]')
        print(hist_str(out.flatten().float()))

        # bright fraction
        for thr in (0.3, 0.5, 0.7, 0.9, 0.95):
            frac = (out > thr).float().mean().item()
            print(f'  fraction > {thr}: {frac*100:.2f}%')

        # in the "lit" regions (output > 0.3), what's the mean brightness?
        bright = out > 0.3
        if bright.sum() > 0:
            mean_bright = out[bright].mean().item()
            print(f'  mean brightness in lit pixels (out>0.3): {mean_bright:.3f}')
        very_bright = out > 0.9
        if very_bright.sum() > 0:
            mean_vb = out[very_bright].mean().item()
            print(f'  mean brightness in very-bright pixels (out>0.9): {mean_vb:.3f}')


if __name__ == '__main__':
    main()
