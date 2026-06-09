# W2F 夜间自主接管交接文档（给新 session）

> 你是接管 W2F 项目夜间自主迭代的 Claude Code session。用户睡了，全程自主，攒结果等用户醒。**中文回复。**

---

## 0. 任务 & 目标
- **W2F (Words-to-Faces)**：把 K=12 个**字母 patch** 经 STN 摆位、合成成一张**人脸线稿**（CelebA-edge 风格）。用 **drifting loss**：D1=每个字母 drift（在 pixel 空间，对真字母 bank）、D2=整张人脸 composite drift（对真 edge-face bank）。
- **🎯 目标：edge-domain KID 刷到 ≤ 75**（越低越好）。当前**诚实最好 = 152.6**（sweep41_aggbase_w03, dw0.3）。纯脸天花板 ~35（sweep13）。

## 1. 🚫 铁律（绝不能破）
1. **不作弊**：禁 per-pixel alpha（逐像素强度=第二幅自由的画，作弊）；禁 anchor（手写五官坐标=喂人脸先验）。**合法合成只有：sum+clamp，或 per-slot 统一强度 `--slot-intensity`（每层一个标量增益）。**
2. **无人脸先验**：不能用"为人脸专门设计的 kernel"（生成前期没有人脸特征）。**通用结构核（IoU/dice/gradient）可以**。
3. **字母 loss：D1 drift 为主**；**CE 分类器只能当小辅助**（survey 已证：裸 classifier-CE 当 loss 会出对抗样本/被钻空子；只能在 drift 之上加小权重，最好配鲁棒 CNN）。
4. **aggressive 增广 + 32px patch**。
5. **commit-before-run**：launch 任何 run 前先 `git commit` 代码快照。维护 git，备份必要 ckpt（别丢 AE！）。
6. **大文件存数据盘**（见下）。

## 2. 🛠 基建（最重要，照抄）
- **机器**：3 卡 AutoDL RTX PRO 6000（96GB×3）。
  `ssh -p 15455 root@<SSH_HOST>`  密码 `<SSH_PASSWORD>`
- **askpass**（WSL 里）：`/root/askpass.sh` 内容是 `echo <SSH_PASSWORD>`。若不存在，在 WSL 重建：
  ```
  printf '#!/bin/bash\necho <SSH_PASSWORD>\n' > /root/askpass.sh && chmod +x /root/askpass.sh
  ```
- **跑 SSH 命令的固定模板**（验证可用）：
  ```
  wsl -- bash -c 'export SSH_ASKPASS=/root/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 && ssh -p 15455 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 root@<SSH_HOST> "远端命令"' 2>&1 | tr -d '\0'
  ```
  - 用 `-p`（ssh）；`SSH_ASKPASS_REQUIRE=force` 免 setsid；**末尾 `tr -d '\0'` 去 WSL null 字节**（否则乱码）。
- **★ scp 坏了（AutoDL 的 SFTP 子系统被阉割，plain scp 和 scp -O 都挂）。传文件只用管道**：
  - 上传：`cat 本地文件 | ssh ... "cat > /root/w2f/远端路径"`
  - 下载：`ssh ... "cat /root/w2f/远端文件" > 本地文件`
  - **单文件、别链式、别并发太多**（并发多了连接会堵；堵了就 TaskStop 掉卡死的后台任务）。**拉图失败就 skip，训练优先。**
- **★ 磁盘**：系统盘 `/` 30G **只剩 ~7G**；数据盘 `/root/autodl-tmp` **120G 空**。**所有新 run 的 samples/logs/ckpt 都写 `/root/autodl-tmp`**（launch 脚本已默认）。代码在 `/root/w2f`（系统盘）。
- **远端 python**：`/root/miniconda3/bin/python`。
- **SSH 命令坑**：① 别用圆括号 `()`（破 shell）；② 别在 `bash -c '...'` 里再嵌单引号（用 `;` 分隔、避免引号）；③ **`A && cmd &` 会把整条（含 cd）后台化** → 多命令用 `;` 分隔，或日志用绝对路径；④ 新实例刚开机/迁移时 `kex_exchange_identification` = 没就绪，等 1-2 分钟别硬戳。
- **本地项目路径**：`<LOCAL_REPO_PATH>/w2f`（WSL 下 `/mnt/<drive>/.../w2f`）。注意若路径里含特殊字符的目录名，**全路径内联**、别用 shell 变量装它（会出空值）。

## 3. 📦 关键文件 & checkpoint
- 模型：`src/face_drift_multi_transformer.py`（`HierarchicalSlotGen`：K 槽 token+eps+letter_embed → Stage1 transformer → theta(STN参数)+seed → theta_pe → Stage2 transformer → per-slot CNN decoder → 32×32 patch → STN 贴到 128 画布 → sum+clamp 合成）。
- 启动器：`.local/launch_sweep_agg.sh`，env 参数化：`GPU` `DW`(letter-drift-weight) `NAME` `KERNEL`(l2/iou)。**已默认存数据盘 + sample-every 2000 + 每图层自动存**。用法：
  ```
  ssh ... "cd /root/w2f ; GPU=0 DW=0.1 NAME=sweepNN_xxx KERNEL=iou bash .local/launch_sweep_agg.sh"
  ```
- Eval（都在远端跑，不用传）：
  ```
  python .local/eval_face_kid.py <G_ckpt> checkpoints/face_ae128_metric.pt checkpoints/edge_bank_128_dil_lo75.pt   # → KID/prec/rec
  python .local/eval_letter_metric.py <G_ckpt> checkpoints/letter_cnn_32_mild.pt                                   # → 字母 acc
  ```
- 远端 `/root/w2f/checkpoints/`（**别删别丢**）：`edge_bank_128_dil_lo75.pt`（面 drift bank）、`face_ae128_metric.pt`+`face_lat_metric.pt`（**KID 的 AE，丢过一次，绝不能删**）、`letter_cnn_32_mild.pt`（字母 eval）、`aug_letter_bank_32_aggressive_2000.pt`（aggressive 字母 bank，已转 float）。

## 4. 🔥 成脸甜点区（这套配方才出脸，别乱动 r/rio/overlap）
`K=12 bgen=2048 gen-per=64 r=0.10-0.40 region-iou=1.5 n-regions=16 overlap=5 sharpness=0.5 sum+clamp d-token=320 n-layers=6 patch=32 aggressive-aug drift字母`（= sweep35 v15/vD 的成脸区）。**偏离这个（如 r→0.8、rio→6、overlap→0）脸就散。** launch 脚本已锁这套。

## 5. 当前在跑（启动时间见 log t=）
3 个 run（GPU0/1/2），各 16000 步 ~3.7h，往"低字母权重 + IoU 核"探：
| run | 配方 | log / ckpt |
|---|---|---|
| sweep43_dw01_l2 | dw0.1 + l2 核 | /root/autodl-tmp/{logs/sweep43_dw01_l2.log, samples/sweep43_dw01_l2/} |
| sweep44_dw01_iou | dw0.1 + IoU+topk4/8 | …/sweep44_dw01_iou… |
| sweep45_dw005_iou | dw0.05 + IoU+topk4/8 | …/sweep45_dw005_iou… |
- IoU run grad 偏大(75/190)，盯着别 NaN/inf。bvar 健康 ~0.06-0.09，别塌向 0。

## 6. 🔁 工作流（每轮循环做）
1. **监控**：一条 SSH —— `pgrep -af face_drift_multi_transformer|grep -v grep|wc -l`；`tail -2` 三个 log 看 step/16000、bvar(别塌0)、alive(12/12)、ovl、OOM/Traceback；`nvidia-smi` 三卡。报进度。崩了看 traceback 修（坑见 §7）。
2. **任一 run 跑完**（step16000 / 进程消失 / log 出 `done`）→ eval（§3 两条）→ 报 KID+acc。**KID<152=新 best；KID≤75 → PushNotification 通知用户**。管道拉 `multitrans_step16000.png`+`*_layers.png`+`*_letters.png` 到本地 `w2f/remote_samples/<name>/`（失败 skip）。`cp` best ckpt 到 `/root/autodl-tmp/backup_runs/`。
3. **GPU 空 → 上下一批**（见 §8）。commit-before-run，管道推改动的代码，py_compile 验证，再 launch。
4. 攒 viz + 结果，等用户醒来汇总。

## 7. 已知坑 & 解法
- **uint8 bank 崩 `normal_kernel_cuda not implemented for Byte`** → 把 bank 转 float 重存：`python -c "import torch,sys;b=torch.load(sys.argv[1]);b=b.float()/255.0 if b.dtype==torch.uint8 else b;torch.save(b,sys.argv[1])" <bank.pt>`（aggressive bank 已转好）。
- **`logs/... No such file or directory`** → cd 被 `&` 后台化，用 `;` 分隔、日志用绝对路径。
- **eval 报 `unexpected keyword 'slot_intensity'`** → 远端 model 是旧版，管道推当前 `src/face_drift_multi_transformer.py` 上去（含 slot_intensity + 每图层 viz）。
- **scp 挂** → 用管道（§2）。
- **后台 SSH 卡死** → TaskStop 掉，单条干净重来。

## 8. 下一批实验（往 KID≤75 推，按优先级）
1. **★ FiLM 字母条件（方向①，最重要）**：现在字母是"相加"进 token，条件太弱+纠缠。改 `src/face_drift_multi_transformer.py`：letter_embed → 小 MLP → 每通道 (γ,β)，调制 decoder 特征 `h=γ*h+β`；加 `--letter-film` flag（默认 off，向后兼容）。改完 commit→管道推→py_compile→launch（基底=成脸区+aggressive+KERNEL=iou+DW=0.1+--letter-film）。**字母可以影响布局没事，问题是条件强度，别去解耦布局。**
2. **更低字母权重** DW=0.03 + iou。
3. **CE 小辅助**：`--letter-cls-weight 0.3`（配 letter_cnn_32_mild，drift 仍主）。
4. 若想换核/objective：先**翻纯脸单图层实验日志**（远端 `/root/w2f/logs/sweep2_*` 到 `sweep28_*`、`drift_*`、`sweep13_*`）找经验；通用核 OK，无人脸先验。

## 9. 可视化（给 report 攒料，已在 `_save_grid` 实现 commit e5dd02a）
每 2000 步存：① `multitrans_stepN.png`（3 行：上=各色图层、**中=模型输出**、下=真脸数据集；★判断脸只看中间行）+ 每列标该样本的 12 个字母；② `_letters.png`（每槽 decoder patch + 目标字母对照）；③ `_layers.npy`+`_layers.png`（每图层单独备份）。都在数据盘。目标多攒 viz。

## 10. git
本地 git，commit-before-run。近期 commit：`57deba5`(3卡launcher)、`e5dd02a`(每图层viz)、`f72f5b2`(uint8修复)、`564cd89`(slot_intensity)。改代码先 commit 再 push 再 launch。

---
**第一步就做**：按 §6 监控当前 3 个 run，报进度；确认连得上、3 个都在跑。然后照工作流循环到用户醒。
