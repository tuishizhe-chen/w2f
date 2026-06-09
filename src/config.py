"""Central config for all hyperparameters.

v1: baseline — D1 drift loss in pixel space, no D2.
v2: D1 in 16x16 avg-pooled feature space; add pairwise letter repulsion; wider pos range; K=5.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # data
    letter_size: int = 32
    canvas_size: int = 128
    K: int = 5  # letters per string (v2: 4->5; more "face-like" layout room)
    batch_strings: int = 32  # B
    pos_per_letter: int = 8  # positive affine samples per letter
    # v2: drift loss operates on down-sampled feature space (avoid pixel-space curse)
    drift_pool: int = 8   # avg_pool kernel → 128/8 = 16×16 = 256-dim

    # model
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    mlp_ratio: float = 4.0
    n_letter_classes: int = 26  # A-Z
    init_scale: float = 0.5
    init_trans: float = 0.0

    # drift loss
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2)
    d1_weight: float = 0.3
    d2_weight: float = 0.2    # v8: back to v4 value (v7's 2.0 hurt acc)
    # v3: classifier-guided letter fidelity (cross-entropy on inverse-STN crops)
    cls_weight: float = 1.0
    # pairwise letter-center repulsion
    # v2: 0.3, 0.35;  v4: 2.0, 0.45;  v6: dual-margin (stricter for same-class)
    repulse_weight: float = 2.0
    repulse_margin: float = 0.45
    repulse_margin_sameclass: float = 0.80  # much stricter for duplicate letters
    # v4: diversity loss — penalize if different eps → same canvas
    diversity_weight: float = 0.5

    # train
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    steps: int = 12000  # v6: +2k for more convergence
    log_every: int = 100
    ckpt_every: int = 1000
    wall_clock_min: float = 75.0  # minutes, leaves 15min for eval
    seed: int = 42

    # data prior (v2 widened)
    pos_scale_range: Tuple[float, float] = (0.18, 0.40)   # random positive scale
    pos_trans_range: Tuple[float, float] = (-0.55, 0.55)  # random positive translation

    # eval strings
    test_strings: Tuple[str, ...] = ("ABCD", "HELO", "FACE", "XYZW",
                                    "YUXU", "AIAI", "HIHI", "OKOK")
    eval_samples_per_string: int = 4  # different noise samples per prompt


def get_cfg() -> Config:
    return Config()
