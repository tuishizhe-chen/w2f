# Words to Faces (w2f) — Project Spec

## 项目目标（来自 Yuxuan 的 slides `w2f - Google Slides.pdf`）

**输入一串字母 (e.g. "HELLO")，一步生成一张图像，使之：**
1. **整体像一张人脸**（face distribution match，proxy: CelebA）
2. **每个字母仍然可辨识**，它们分布在图像中不同位置、经过 scale/translation（可选 rotation/shear），像人脸五官的位置摆布。

可识别的"成功案例"：肉眼可见是人脸轮廓，并且能从图里辨认出所输入的字符。

## 核心方法（来自 slides + arXiv 2602.04770 "Generative Modeling via Drifting"）

### 1. Generator `G(x1,…,xk, ε) -> y1,…,yk`
- `xi` 是第 i 个字母的 image (32×32)
- `yi` = STN(xi; θi) — 字母在 128×128 画布上的 affine 变换（scale + translation，可选旋转）
- **backbone**：MLP 编码每个字母 + noise token ε → Transformer → 为每个字母输出 affine 参数 θi
- 最终图 = 所有 yi 的 max（或 sum-clip）合并
- **一步生成**（无迭代）

### 2. 两个监督信号（无 GAN）
- **D1 (letter fidelity)**：每个生成 patch 应仍是其对应字母
  - Drifting loss：`L_D1 = ||y_i - sg(y_i + V_p(x_i, y_i) - V_q(y_i))||^2`
  - `V_p⁺` 吸引到 "字母 x_i 在画布上的合法变换分布"（正样本：真实字母的随机 affine 采样）
  - `V_q⁻` 排斥同 batch 中同字母的其他生成样本（防 mode collapse）
- **D2 (face distribution)**：整张图应像人脸
  - Drifting loss 在 **MAE latent space**（像素空间高维不 work，slides 明确说）
  - MAE 先在 face 数据上 SSL 预训练

### 3. Drifting kernel 细节（来自 lambertae/drifting 仓库 `drift_loss.py`）
- 多温度 `R_list = (0.02, 0.05, 0.2)`，每个 R 独立归一化 force 再相加
- `k(x,y) = exp(-||x-y||/τ)`，用 softmax 在 batch 维做归一化（对比学习形式）
- 自对角屏蔽（diag mask + large penalty）
- 每 R 的 force：`r_coeff = aff_pos * Σ aff_neg − aff_neg * Σ aff_pos`
- Loss：`mean((gen_scaled − sg(gen_scaled + V))²)`

## 数据
- **字母**：torchvision.datasets.EMNIST(split='letters')，26 类，28×28 → resize 到 32×32
- **人脸**：CelebA (若网络不通，用 torchvision LFWPeople，再不行合成 proxy — Mixture-of-Gaussians 椭圆+五官 mock，保证 pipeline 可跑通)
- 分辨率：letter 32×32，canvas 128×128

## 架构尺寸（slides "Architecture design"）
- Letter space: 32×32（最多 48×48）
- Face space: 128×128
- MLP → Transformer (6~8 layers, d=384) → per-letter θ∈ℝ⁶ (affine)
- STN 支持 scale/translation（初始化 scale=0.5, translation=0 at epoch 0 → drift）

## 文件结构
```
w2f/
├── PROJECT.md            本文件
├── EVAL_CRITERIA.md      成功/失败判定
├── ITERATION_LOG.md      每版迭代日志（agent 自己写）
├── STARTUP_PROMPT.md     启动提示词（给 agent 的完整指令）
├── src/
│   ├── config.py         所有超参
│   ├── data.py           EMNIST / CelebA 加载 + augment
│   ├── model.py          Generator (MLP+Transformer+STN)
│   ├── mae.py            face MAE encoder (预训练 + 冻结)
│   ├── drift_loss.py     PyTorch port of lambertae/drifting
│   ├── classifier.py     letter classifier (评估用)
│   ├── train.py          主训练循环
│   ├── evaluate.py       评估脚本（输出 metrics + 样本图）
│   └── pretrain_mae.py   MAE on face data
├── data/                 数据集缓存
├── checkpoints/v{N}/     每版 checkpoint
├── samples/v{N}/         每版生成样本
├── logs/v{N}.log         训练日志
└── drifting_ref/         参考实现（已 clone）
```

## 迭代策略
- **v1 (baseline)**：STN + Transformer + 双 drifting loss，MAE 用轻量 (ViT-Tiny)
- 每版跑完后 agent 自动：
  1. 生成 8 个测试字符串的样本网格
  2. 用预训练 letter classifier 在生成图的 STN patch 上算 letter accuracy
  3. 算 face 特征与 CelebA 的 Fréchet 距离（MAE latent 上）
  4. 写 ITERATION_LOG.md，总结失败模式，提出 v{N+1} 方案
  5. 不需询问用户，直接进入下一版

## 评估标准（详见 EVAL_CRITERIA.md）
- 终极成功：letter_acc ≥ 0.7 且人类视觉能看出"是人脸"
- 软性中间目标：每版至少有一项 metric 相对上一版有可观测改进，否则判定为"需要根本性改方案"
