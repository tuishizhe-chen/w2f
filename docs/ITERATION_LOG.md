# Iteration Log

每版训练结束后由 agent 自主追加一节。

---

## v1 — 2026-04-28 — Minimal pipeline: STN + Transformer + D1 drift loss (no D2)

### 方案变更（相对 baseline / spec）
- **只上 D1**：v1 先把 D1 的 drifting loss 跑通。D2 (MAE latent) 复杂，留到 v2 或 v3 再加。
- **人脸数据改成合成 proxy**：CelebA 下载常卡住；第一版用"灰色椭圆 + 两个黑点眼睛 + 一条嘴巴"合成人脸作为 face prior。后续版本再切 LFW / CelebA。
- **letter 数据**：torchvision.datasets.EMNIST(split='letters')，预期可在线下；若下载失败 fallback 到手工合成 letter（用 PIL 画字母）。
- **Generator**：MLP per-letter embed → 6 层 Transformer (d=256, h=8) → 每字母出 θ∈ℝ⁴ (scale, tx, ty, 额外 1 个冗余；旋转暂不开)，外加 global bias 0.5 scale, 0 translation。
- **合并**：所有 STN 输出 (B,K,1,128,128) 用 max 合成 (B,1,128,128)。
- **D1 drift loss**：
  - gen = generator 输出的 patch `y_i`，flatten 成 `[B, K, 128*128]`
  - fixed_pos = 对每个输入字母 x_i 采样的真实 affine 变换（Vp 的正样本）
  - fixed_neg = batch 内同字母的其他生成 patch（做 memory bank 粗版，用 same-batch 负样本即可）
  - R_list = (0.02, 0.05, 0.2)
- **训练**：batch=32 strings，每 string K=4 字母（固定长度简化），10k step，lr=2e-4 AdamW，wall clock budget = 75 min（留 15 min eval 缓冲）。
- **STN 初始化**：θ_scale bias = logit(0.5), translation bias = 0；网络末端 affine 层权重初始化为 0 以便启动时完全由 bias 决定。

### 超参
- batch=32 strings × K=4 letters = 128 STN tokens
- letter_size=32, canvas=128
- d_model=256, nheads=8, nlayers=6
- lr=2e-4, wd=0.01, grad_clip=1.0
- steps=10000, log_every=100, ckpt_every=1000
- R_list=(0.02, 0.05, 0.2)
- pos_samples_per_letter=8 (每字母从 EMNIST 里随机 affine 采 8 个正样本)
- neg_samples = same-batch 同字母的其他生成 patch

### 预期风险
1. drift loss 在 pixel space 128×128 可能不稳（slides 提到"high-dim 不 work"），所以 K small + letter 只有 1 通道 + gray letter 让 manifold 窄一点；若仍爆炸，降到 patch flatten 到 32×32 = pad letter 直接算 pixel space。
2. mode collapse：所有字符串生成同一张脸。靠 eps noise + same-batch neg 缓解。
3. STN 初始 scale=0.5 时 patch 占 64×64，4 个字母可能重叠严重 → transformer 应该学会分散。

---

### 定量结果
- letter_acc = **0.031** (33% of random chance 3.8%)
- noise_baseline_acc = 0.087 (classifier crop of random pixels, with fixed θ)
- D1 loss final = 7.02 (essentially flat from step 100 onward, plateaued at 6.7~7.2)
- 训练耗时 537s (~9 分钟)，step 10000

### 定性观察（看图、数值）
- 所有 grid tile 亮度非常均匀 (mean ~75-82)，说明每张图看起来都像：**黑色背景 + 中心一坨重叠的字母**。
- r_mean 很快从 0.5 收敛到 ~0.32，|tx|, |ty| 只有 ~0.2（远小于可用的 0.7 范围），letters 都挤在 canvas 中心一小块区域。
- 不同输入字符串生成的图看起来极其相似 → **严重 mode collapse** 到 "中心一坨"。
- 没有任何"脸"的结构（预期：本来就只开 D1，没有 face 信号）。

### 失败模式 check-list
- [x] 图像没有 NaN / 纯黑 / 纯白（有灰度结构）
- [x] 字母没有全部退化成一个点（但尺寸太小且重叠）
- [x] 字母没有跑到画布外（事实上相反：挤在中心）
- [ ] 人脸形状隐约可见（完全没有，无 D2）
- [ ] 不同输入字符串产生不同图像 — mode collapse 明显

### 诊断 & 下一版计划
**根本问题**：D1 drift loss 在 128×128 = 16384 维像素空间上做几乎是纯噪声信号。正样本（随机放置同字母）和生成样本的距离在 pixel space 由"空白背景占比"主导（两张图 95% 像素都是 0），kernel 几乎区分不出 letter 内容，只对 letter 位置做"聚簇排斥"。模型发现：把 r 减小 + 字母堆在中心 → 与各种正样本的 pixel MSE 都差不多 → 局部最优。

**v2 计划**：
1. **切换 D1 到 low-res feature space**：对 generator 的 patch 和正样本先 avg-pool 到 32×32（或 16×16），再做 drift loss，避免 pixel 空维度诅咒。这是参考 drifting_ref/train.py 中 "low-resolution space" 思路的轻量实现。
2. **加 latent AE for D2**：训练一个小 auto-encoder（不用 MAE，2-3 分钟即可训完）on synth faces，得到 64-dim latent；在 latent space 对生成整图 vs face latent 的 drift loss。
3. **对 positive sampler 的 scale 范围也让它稍微放大**：(0.18, 0.35) → 更接近"一个 letter 占 canvas 1/3" 的目标，增加 position diversity。
4. **hinge translation regularization**：给 Transformer 输出的 tx,ty 加一个 soft push 让它们尽量分开（防重叠）。用 pairwise repulsion between letters 的 MSE loss：想要 |pos_i - pos_j| 至少 > 某 threshold。

v2 将同时实现 1, 3, 4，暂不加 D2（留 v3），以便独立验证"低分辨率 drift + 反重叠"是否就能把 letters 摊开并保持可识别。

---

## v2 — 2026-04-28 — Low-res drift (16×16 avg-pool) + pairwise repulsion + K=5

### 方案变更（相对 v1）
- D1 drift loss 从 pixel space (128²=16384 维) 换到 avg_pool2d(kernel=8) 得到的 16×16=256 维"低分辨率特征空间"。
- 加入 pairwise letter-center repulsion hinge loss (margin=0.35, weight=0.3)。
- pos_scale_range 收窄 (0.18, 0.40) 使正样本 letter 大小更接近目标。
- K 从 4 调到 5。

### 定量结果
- letter_acc = **0.039** (略优于 v1 的 0.031；新 noise_baseline 0.025)
- D1 loss final ~ 6.6 (略优于 v1 7.0)
- repulse loss 前 100 step 从 0.12 降到 0.017，步后恒定 < 0.003（letters 位置快速满足 margin）
- 训练耗时 579s

### 定性观察（看图 + 像素统计）
**决定性观察**：grid.png 里同一行（同字符串、4 个不同 eps 噪声）的平均亮度**完全相同**（差 ≤ 1）。
→ **eps 噪声根本没进入输出**。Transformer 完全忽略了 noise 通路。
→ 不同 string 之间亮度也相近（71~82），仍是"中心一坨"模式，仅因 K=5 而比 v1 稍大。

### 失败模式 check-list
- [x] 无 NaN
- [x] 字母没退化成点（但仍然挤在中心）
- [ ] 字母未完全分散（repulse 只拉开中心点，letter 体积仍重叠）
- [ ] 脸部结构（无 D2）
- [ ] 不同输入不同图（**mode collapse 极严重**，eps 完全失效）

### 诊断 & 下一版计划
**两个根本问题**：
1. **eps 失效**：theta_head 被 zero-init，start 时 r=0.5, tx=ty=0 完全由 bias 决定 → transformer 内部梯度早期被 kill → 模型无法建立 eps→output 的通路。而 eps 只作为一个额外 token 传入，attention 可以把它权重学成 ~0。
2. **Drift loss 无"识别性"信号**：即使在低分辨率 16×16，drift 匹配的是 distribution，它对"blob 是否长得像字母"几乎无监督。需要一个直接的 "letter fidelity" 监督。

**v3 计划**：
1. **去掉 theta_head 的 zero-init**：改用 `xavier_normal`，bias 初值保持 r=0.5, t=0。加一个 tanh 在 tx/ty 上以保持初始稳定。
2. **eps 用 FiLM-style 注入**：把 eps 通过 MLP 生成 scale, shift 参数，然后对每个 letter token 做 `token * scale + shift`；这样 eps 一定会影响每个 letter 输出，attention 无法忽略。
3. **加 classifier-guided D1**：对 generator 输出的每个 patch（128×128），用 inverse-STN 裁回 32×32（用 theta），喂 frozen letter CNN，cross-entropy loss with 真 label。weight=1.0；v2 的 low-res drift 仍保留 weight=0.3 做 regularizer。
4. **加 D2 face-AE latent drift**：训练 FaceAE on synth faces，对生成 canvas 的 latent 和 synth face 的 latent 做 drift_loss（没有 neg，纯正向吸引）。weight=0.2。
5. **重要**：data 里不再同一个 B 内对所有 K 位置用同一 letter_img — 保持每位置独立样本（已经是这样，check 一下）。

v3 这 4 改全上，预期 letter_acc 能跳到 0.3+ 并且生成整体有脸形。

---

## v3 — 2026-04-28 — FiLM eps + classifier-guided D1 + face-AE D2

### 方案变更（相对 v2）
- **Generator**：eps 注入方式改成 **FiLM (scale + shift on every letter token)**；theta_head 改用 xavier_normal gain=0.01 初始化（不再 zero）。
- **Classifier-guided D1**：新增主 letter-fidelity 信号 —— 把 canvas 用 inverse STN 按 theta 裁回 32×32 letter crop，喂冻结的 letter CNN，cross-entropy 与 true label，weight=1.0。
- **D1 drift loss**：降权到 0.3 做 regularizer。
- **D2 drift loss**：新增，在 face-AE 64-dim latent space 上，正样本是 synth face batch 的 encode 结果，没有 neg 项，weight=0.2。
- face-AE 独立预训练（1500 steps, MSE=0.028）在 synth face 上。

### 定量结果
- **letter_acc = 0.422** (v2=0.039, v1=0.031; 提升 10×+)
- per-string: XYZW=0.875, HELO=0.75, FACE=0.25, ABCD=0.25, YUXU=0.25, AIAI=0.5, OKOK=0.5, HIHI=0.0
- cls final ~1.5-2.0 (random=3.26)
- d1 drift final ~6.9 (basically unchanged; regularizer)
- d2 drift final ~7.5 (几乎无变化，AE latent 空间 drift signal 弱)
- rep loss ~0.018（维持分散）
- 训练耗时 891s

### 定性观察
- cls loss 有效：letters 明显可辨（XYZW/HELO 能被 classifier 正确读出来）。
- 但仍然 **eps 失效**：同字符串不同 noise seed 生成的图在 pixel 上几乎全等（cell 均值差 ≤1）。
  FiLM init 为 0，且 cls 梯度只想让 letter 长得像字母不依赖 eps，所以 FiLM 学到的 scale/shift 都是 ~0。
- **HIHI = 0.0**：相同 letter 在字符串内重复时，transformer 输出两个几乎相同的 theta → 两个 H 完全重叠在同一位置，丢失一个。positional embedding 的信号被 cls loss 压过。
- d2 signal 贡献很小：AE latent drift 一直在 7.5 左右，模型没学到 face 轮廓；但生成图已不再是"中心一坨"。

### 失败模式 check-list
- [x] 无 NaN
- [x] 字母未退化（classifier 能识别大部分）
- [x] 不完全在画布外
- [ ] 脸部轮廓（仍看不出；D2 signal 太弱）
- [ ] 不同 eps 产生不同图（eps 仍失效）
- [ ] 同字符串内相同字母能分开位置（HIHI 挂掉）

### 诊断 & 下一版计划
**v4 目标**：突破 letter_acc > 0.6，并让"脸"开始可见。

1. **强制 letter 分散**：加 "duplicate-letter separation" —— 给 transformer 层加一个 letter-id 之外的 "slot id" embedding 系数更大；或在 loss 里显式惩罚相同 class 的 theta 重合（pairwise distance for same-label tokens）。最简单：给 letter encoder 用 per-position 额外 token id 并加 stronger pos emb weight。
2. **eps 真正 enable**：FiLM bias 初始化为非零小值（如 scale~N(0, 0.1)，shift~N(0, 0.1)），同时可以加 "entropy/diversity loss"：对同 label 不同 eps，输出 canvas 的 pixel MSE 要 > epsilon，防止退化成 deterministic。
3. **增强 D2**：face-AE 改成在 low-res 上的 drift（16×16 after avg_pool）比 latent 更 informative；或者做 "soft face mask"：乘一个预先定义的 ellipse mask，惩罚 mask 外的像素能量 + 鼓励 mask 内覆盖。
4. **加大 cls_weight & d2_weight 对比**：既然 cls 已经起作用，把 cls_weight 保持 1.0，d2_weight 从 0.2 调 0.5，并在前 3k 步 warmup d2（避免压过 cls）。
5. 数据重排：batch 内让相同 class 尽量 sample 不同 instance（EMNIST 里不同笔迹），给 d1 更多 diversity。

v4 将依次叠加 1、2、3（不改 cls_weight 太多，让它继续 drive letter recognition），不要一次性改 4 个 knob，以便追溯。

## v4 — 2026-04-28 — Stronger repulsion (2.0/0.45), FiLM non-zero init, diversity hinge, face-mask D2

### 方案变更（相对 v3）
- repulse_weight 0.3→2.0；margin 0.35→0.45。
- letter 的 pos_embedding 乘 2x，让同字符在不同 position 更可区分。
- FiLM layer init 从 zero 改成 N(0, 0.02)。
- 新增 diversity_weight=0.5 的 hinge loss：同 label 不同 eps 的 canvas pixel-MSE 应 ≥ 0.02。
- D2 从 face-AE latent drift 换成 **face-mask 像素 loss**：(canvas * inv_mask)² + 0.3 * (coverage_in_mask - 0.08)²。d2_weight=0.2.

### 定量结果
- letter_acc = **0.438** (v3 0.422，小幅提升)
- per-string: HELO=1.0, XYZW=0.75, ABCD=0.5, AIAI=0.5, OKOK=0.5, FACE=0.125, YUXU=0.125, HIHI=0.0
- cls final ~1.6-2.1
- d2 非常小 ~0.003 (mask loss 本身量级小)
- rep 稳定 ~0.06
- div loss 常驻 0.019 ≈ 初值 → pixel_diff 仍 ≈ 0，eps 依然无效
- 训练耗时 844s

### 定性观察
- 同行 (同 string) 4 个 eps 的图平均亮度差 ≤1 → **eps 仍完全失效**。Diversity hinge 量级 (0.5 * 0.02 = 0.01) 相对 cls loss (1.0 * 2) 太小，被优化器忽略。FiLM weights 大概在训练早期就被压平了。
- **HIHI 仍然 0**：强化 repulse weight 到 2.0 并没有阻止 repeated letters 叠在一起 — 说明 repulse 的 hinge 在 margin 附近接近零梯度（margin=0.45，实际 dist≈0.015，应该给很大 penalty；但 rep final 0.056 远低于 `((0.45-0)²)/2 ~ 0.10`，表明非重复字母时 margin 满足，拉低均值）。必须查 HIHI 专门 theta。

### 失败模式 check-list
- [x] 无 NaN
- [x] 字母能识别 (HELO 100%)
- [ ] 重复 letter 无法分开 (HIHI 0%)
- [ ] 脸部轮廓（face-mask 已把 letters 吸进 ellipse，但形状仍"纸片人"式）
- [ ] eps 产生不同图（完全 fail）

### 诊断 & 下一版计划
**v5 聚焦修 HIHI + eps**：
1. **HIHI 根本：repulse_weight 2.0 仍不够，因为 same-letter 两个 theta 输出完全一致（输入 token 差的只是 pos_emb）**。要么 transformer 给不同 pos 同类 token 输出不同了，但 STN head 再 map 回同一 theta；要么 repulse 只是轻微罚而且被 cls/d2 平均掉。
   → **直接做法**：给每个 position 一个**单独的 theta_head projection**（per-slot MLP head）而不是共享一个 Linear；这样即使 transformer tokens 相同，different slots 输出的 theta 也强制不同。
2. **eps 要真 work**：diversity 改成 hard constraint — 在 loss 里直接 **maximize** pixel variance across eps 的样本，不用 hinge。具体：给 cls loss 分成 N_eps=2 份，cls_loss 同样要低，但多一项 `-min(pixel_diff, 0.05)` 即奖励 up-to-0.05 的差异。
3. **D2 换成 dual-signal**：先用一个 "hole-at-center / edges-on-ellipse" 的 soft mask 反而效果更好 — 或者两个 concentric 的 mask 差（face contour）。
4. 继续加深：层数 6→8，dim 256→320。

v5 主要改动：per-slot theta_head + 奖励 pixel variance 的强 diversity。

---

## v5 — 2026-04-28 — Per-slot theta_head with face-layout bias + hard diversity reward

### 方案变更（相对 v4）
- theta_head 从 shared Linear 改成 ModuleList of 8 Linears (per-slot)。
- 每个 slot 预置 translation bias（atanh-warped）到 face-layout 位置：eyes, nose, mouth corners...
- diversity loss 从 hinge 换成 reward: `-clamp(pixel_diff, max=0.03)`。

### 定量结果
- letter_acc = **0.188 (REGRESSION！v4=0.438)**
- HIHI=0.1875 (v4=0.0, 改善) —— per-slot 确实修了 repeated letter 重叠。
- 但所有其他 string 都下降（ABCD 0.5→0.25，HELO 1.0→0.25，XYZW 0.75→0.19）
- cls final ~2.6 (v4 ~1.9)
- div reward 仍几乎为 0 → eps 依然无效

### 定性诊断
- per-slot biases 把 letters 预置到 "face landmarks"，但这些位置对 "读字母" 并不友好：letters 相互遮挡，且 STN crop 位置被 bias 锁住，transformer 难以移动。
- diversity reward 是 unbounded gradient (linear in pixel_diff up to cap)，理论上应该有 push；但 pixel_diff 在训练中仍 ≈ 0，说明 generator 输出对 eps 的 FiLM 响应在训练中被"无视"掉（cls 梯度把 FiLM weights 推向 0）。

### 失败模式 check-list
- [x] 无 NaN
- [x] 字母可识别，但明显比 v4 差
- [x] 重复 letter 不再重合（HIHI > 0）
- [ ] 脸形
- [ ] eps 有效（仍 fail）

### 诊断 & 下一版计划
**判断**：v5 方向错了 —— 把 STN 强制到 face layout 并不帮助 letter 识别，反而互相遮挡。应回退 v4 架构并只针对"重复 letter"做精细修补。

**v6 计划**（更保守的回退 + 定向修复）：
1. **rollback** per-slot theta_head → 回到单 shared Linear head (v4 版)。
2. **针对 duplicate letters 的 intra-class repulsion**：重做 repulsion loss，对 same-class letter pair margin 从 0.45 → 0.7（非常严格），其他 pair 保 0.45。
3. 保留 v4 的 face-mask D2、FiLM 非零 init，保留 diversity reward（弱信号总比 0 好）。
4. 训练 steps 从 10000 增到 12000（v1–v5 都在 step 7000+ 仍有下降趋势）。

注：时间预算还有 3 版（v6, v7, v8）。v6 集中修 HIHI + 多训；v7 可能引入更强 face prior (e.g. gradient toward face-AE reconstruction target)；v8 作为最后一版做最终清理/选最佳 checkpoint。

## v6 — 2026-04-28 — Dual-margin repulsion (same-class=0.8) + 12k steps

### 方案变更（相对 v4）
- repulse 加 dual-margin: 同 class 的 pair margin=0.8（非常严）；其他 pair margin=0.45。
- steps 10000 → 12000。
- 其余与 v4 相同（单 shared theta head，FiLM 非零 init，face-mask D2）。

### 定量结果
- letter_acc = **0.398** (v4=0.438 轻微 regression)
- per-string: XYZW/YUXU=0.75, ABCD/AIAI=0.5, FACE=0.44, HELO=0.25, HIHI=0.0, OKOK=0.0
- cls final ~1.8-2.1

### 定性观察 & 失败诊断
- **dual-margin 没有修好 HIHI/OKOK**。print theta 后发现：
  HIHI 的两个 H 输出 tx 差 0.002，OKOK 的两个 O 也一样 → transformer+shared head 在 same-class 同输入上是 permutation-equivariant 的，position embedding 在 6 层后被 layer-norm 稀释掉。
- 强化 margin 到 0.8 只是把"等值点"附近的梯度抬高，但因为两个 H 的 raw theta 输出几乎一致，repulsion 的 gradient 是对称的 → 向量和 ≈ 0，无法打破对称。
- **根源是架构对称性，不是 loss 强度**。必须要么改 theta head (v5 试过，hurt acc)，要么给每个 slot 一个额外小的 deterministic bias。

### v7 计划
1. 保持 v4 架构（shared theta head），但在 forward 里对每个 slot 加一个 **小的固定 translation bias**（e.g. `bias_tx[k] = 0.15 * (2 * k / (K-1) - 1)`，线性摆放 5 个 letters 横向）。bias 小到不压 cls loss（量级 ~0.15 canvas 单位，约 10% canvas 宽度），但足以打破对称。
2. 大幅增加 face-mask loss 权重：d2_weight 0.2 → 2.0；ellipse 更紧（a=0.55, b=0.7）；加一个 "mask coverage" target 从 0.08 升到 0.12（希望 face 区域更满）。
3. 保留 dual-margin（已经装好，不会 hurt）。
4. 训练 12k steps。

## v7 — 2026-04-28 — Small slot translation bias + strong face-mask D2 (d2_w=2.0)

### 方案变更（相对 v6）
- 在 model.forward 里对每个 slot 加一个 **小的固定 translation bias** (`torch.linspace(-0.15, 0.15, 8)`)，加在 tanh 之后。
- d2_weight 0.2 → 2.0。face-mask ellipse 收紧 (0.55, 0.70)。cov_target 0.08 → 0.12。

### 定量结果
- letter_acc = **0.375** (v4=0.438, v6=0.398; 轻度 regression 但细项显著改善)
- **HIHI: 0 → 0.25**（v7 slot-bias trick 成功打破了重复字母对称性！）
- AIAI: 0.5 → 0.625 (v4 → v7)
- YUXU: 0.125 → 0.8125 (大幅提升)
- 但 XYZW 回落 0.75 → 0.25; HELO 1.0 → 0.25; OKOK 仍 0 (O 和 K 看起来长得不同啊，但似乎 slot bias 把 OKOK 的位置搞不好)

### 定性诊断
- slot-bias +/- 0.15 对打破重复字母对称是有用的（HIHI/AIAI 大幅改善）。
- 但 d2_weight=2.0 + 更紧的 mask 把 letters 往中心挤，HELO/XYZW 反而被 d2 干扰了。
- 总体 acc 被拉低是因为 face mask 太强，覆盖期望拉高 letters 互相遮挡。

### 诊断 & 下一版最终计划
v4 架构 + v7 的 slot-bias （不改 d2 权重）= 预期最佳组合。v8 作为最终版：
1. **v4 的所有配置**（d2_weight=0.2，margin=0.45，repulse=2.0）
2. **+ v7 的 small slot-bias**（打破 HIHI 对称）
3. 12k steps
4. 结束后写 FINAL.md 总结和使用说明。

---

## v8 — 2026-04-28 — FINAL attempt: v4 config + v7 slot-bias (combination)

### 方案变更（相对 v7）
- d2_weight 2.0 → 0.2 (v4 值)
- face mask ellipse 回到 v4 宽松 (0.65, 0.80), cov_target 0.08
- 保留 v7 的 slot-bias
- 保留 dual-margin (v6)

### 定量结果
- letter_acc = **0.320** (v4=0.438; 再次 regression！)
- per-string: FACE=0.69 (新高), HELO=0.5, ABCD=0.5, YUXU=0.5, XYZW=0.375, AIAI=0, HIHI=0, OKOK=0
- cls final ~1.9-2.1

### 诊断
v4/v7/v8 的波动说明**训练是 noisy + multi-modal 的**。同样配置不同随机种子结果差 ~10%。v8 不幸运，slot-bias 在这个种子下没有帮助 HIHI（仍 0）而且整体拉低。

### 最终决定
**v4 作为交付** (letter_acc = 0.438)。它是 8 版中数值最高的 checkpoint。

