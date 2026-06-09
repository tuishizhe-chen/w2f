# Evaluation Criteria

每个版本训练结束后，**自动**运行 `src/evaluate.py` 产出以下指标。

## 1. 定量指标

### 1.1 Letter Recognizability (letter_acc)
- 预训练一个 EMNIST letters classifier（简单 CNN，应该 >95% test acc）
- 对于生成图，从 STN θ 反推每个字母的 crop，喂 classifier，算 top-1 accuracy
- **目标：letter_acc ≥ 0.7**（初始可能 <0.1，渐进提升）

### 1.2 Face-likeness (face_score)
- 在 face MAE 的 latent space 上，算生成图 latent 的均值/协方差与 CelebA 验证集的 Fréchet 距离 (FID-lite)
- **目标：face_score 持续下降，相对"noise baseline"至少下降 50%**

### 1.3 Loss curves
- D1 loss, D2 loss, total loss 是否收敛无 NaN

## 2. 定性指标

### 2.1 视觉 sanity check（自动存图）
每版训练结束后，对 8 个固定测试字符串生成 grid：
```
["ABC", "HELLO", "WORLD", "YUXUAN", "FACE", "HI", "AI", "XYZW"]
```
存到 `samples/v{N}/grid.png`

Agent 需要**自己**读图（调用 Read 的图像模式）并在 ITERATION_LOG 里描述观察到的现象，例如：
- 是否一片糊
- 字母是否被"吸"在一起成一坨
- 整体是否已有"脸"的轮廓
- mode collapse（所有字符串看起来一样）

### 2.2 失败模式 check-list
Agent 在每版日志里必须逐条回答：
- [ ] 图像没有 NaN / 纯黑 / 纯白
- [ ] 字母没有全部退化成一个点
- [ ] 字母没有跑到画布外
- [ ] 人脸形状隐约可见（有椭圆轮廓/上下对称）
- [ ] 不同输入字符串产生不同图像（非 mode collapse）

## 3. 迭代终止条件

- **成功**：letter_acc ≥ 0.7 AND face_score 显著低于 noise baseline AND 视觉 pass
- **继续迭代**：上述任一未满足
- **最多 8 版**：若 8 版仍未达成，写最终 POSTMORTEM.md 总结根本性问题和未来方向（但仍算交付，不是失败）

## 4. 每版预算

- 训练时间：**单版 ≤ 90 分钟**（用 2 张 H20，GPU 2、3）
- 超时自动 kill 并记录为 "timeout, partial results"
- 数据集下载卡住 >5 分钟自动降级到 proxy 数据

## 5. Regression guard
v{N+1} 完成后若 letter_acc 比 v{N} 下降超过 0.1，记录 regression 并保留 v{N} checkpoint 作为 fallback。
