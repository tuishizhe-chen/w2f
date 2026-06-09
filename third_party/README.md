# third_party — Drifting (reference)

W2F is built on the **Drifting** generative objective. We do **not** vendor a copy
of it here — see the official sources:

- **Code:** https://github.com/lambertae/drifting
- **Paper:** *Generative Modeling via Drifting*, arXiv [2602.04770](https://arxiv.org/abs/2602.04770)
- **Models:** https://huggingface.co/Goodeat/drifting

## What we used from it

Our `src/drift_loss.py` is a PyTorch re-implementation of the Drifting force from
that repository, with the modifications described in the report:

- a modular structural distance (soft IoU / Dice / multi-scale / gradient / low-freq)
  in place of the original Euclidean-only kernel,
- nearest-neighbour (top-k) pruning of the drift target,
- a data-adaptive softmax temperature,
- input-noise smoothing, and
- two-level application (per-glyph **D1** + composite **D2**) with categorical
  same-class repulsion.

The doubly-normalized geometric-mean affinity, the attraction−repulsion force, the
multi-bandwidth aggregation, and the stop-gradient target are taken from the
original work. Please cite the Drifting paper and respect its license when using
this component.

## To run their reference code

```bash
git clone https://github.com/lambertae/drifting third_party/drifting
```

(`third_party/` is otherwise empty in this repository by design.)
