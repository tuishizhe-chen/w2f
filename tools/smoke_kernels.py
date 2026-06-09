import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from face_drift_pixel import (cdist_iou, cdist_gradient, cdist_lowfreq,
                              cdist_patch, cdist_iou_grad)
torch.manual_seed(0)
x = torch.rand(2, 4, 64 * 64)
y = torch.rand(2, 6, 64 * 64)
for name, fn in [("iou", cdist_iou), ("gradient", cdist_gradient),
                 ("lowfreq", cdist_lowfreq), ("patch", cdist_patch),
                 ("iou_grad", cdist_iou_grad)]:
    d = fn(x, y)
    print(name, "shape=", tuple(d.shape), "range=",
          round(d.min().item(), 4), round(d.max().item(), 4))
