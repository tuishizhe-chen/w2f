import cv2, numpy as np, sys

img_path = '/root/w2f/data/celeba/img_align_celeba/000001.jpg'
bgr = cv2.imread(img_path)
bgr = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
edges = cv2.Canny(gray, 40, 110)
print('edges shape:', edges.shape, 'dtype:', edges.dtype, 'sum:', (edges > 0).sum())
n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
print('n_cc:', n_cc)
print('stats shape:', stats.shape, 'dtype:', stats.dtype)
print('AREA col index:', cv2.CC_STAT_AREA)
areas = stats[:, cv2.CC_STAT_AREA]
print('areas (first 20):', areas[:20])
print('areas range:', areas.min(), areas.max())
print('areas >= 4:', (areas[1:] >= 4).sum(), '/', n_cc - 1)
print('areas >= 8:', (areas[1:] >= 8).sum(), '/', n_cc - 1)
