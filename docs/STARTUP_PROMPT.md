# w2f 项目启动提示词（粘到 tmux 里的 Claude Code 会话开头）

> 把下面整段作为第一条 user message 发给 Claude Code（在 `/home/wangyanmohan/Projectdl/w2f` 目录下启动）。
> 启动后**不要再与它交互**，它会自主完成全部迭代。

---

## PROMPT 正文

你现在在 `/home/wangyanmohan/Projectdl/w2f`。这是一个自主研究项目：**Words to Faces (w2f)**，来自 `../w2f - Google Slides.pdf`。你必须完全自主地把它从零实现到收敛，不要问我任何问题。我不会再回复你。

### 必读材料（按顺序）
1. `PROJECT.md` — 项目目标、方法、文件结构
2. `EVAL_CRITERIA.md` — 评估指标和迭代终止条件
3. `ITERATION_LOG.md` — 迭代日志格式（你自己往里写）
4. `../w2f - Google Slides.pdf` — 原始设计（slides）
5. `drifting_ref/drift_loss.py` 和 `drifting_ref/memory_bank.py` — 参考实现（JAX，你需要 port 成 PyTorch）
6. arXiv 2602.04770 "Generative Modeling via Drifting" — 必要时用 WebFetch 查公式细节
7. arXiv 2111.06377 MAE — 若 D2 用 MAE latent 时查

### 环境
- 8× NVIDIA H20-3e，GPU 0、1 被别人占用，**你只用 GPU 2、3**（`CUDA_VISIBLE_DEVICES=2,3`）
- PyTorch 2.9 + CUDA 12.8 已装，Python 在 `/shared/Applications/miniconda3/bin/python3`
- 开始前跑 `pip install --user torchvision einops tqdm matplotlib scipy` 补齐依赖（如已装则跳过）
- 网络可能时好时坏；torchvision datasets 若下载失败必须有 proxy 降级（PROJECT.md 已规定）

### 自主工作流（每版必走一遍）

对 `v1, v2, …` 循环执行以下步骤，**不要等我确认**：

1. **规划**：在 ITERATION_LOG.md 追加一节，写明本版相对上一版的改动和假设。
2. **实现**：写/改 `src/*.py`。每个 .py 顶部写本版的改动摘要。
3. **Smoke test**：跑 50 step（`--smoke`），确保不 NaN、loss 有限、图能存下来。NaN 就先 debug 到稳定为止。
4. **完整训练**：后台跑 `python src/train.py --version N`，用 `run_in_background=true`，最多 90 分钟；同时用 Monitor 盯 stdout，关键词：`loss=`, `NaN`, `Error`, `Traceback`, `step=`, `done`。
5. **评估**：训练结束后跑 `python src/evaluate.py --version N`，产出 `samples/v{N}/grid.png`、`samples/v{N}/metrics.json`。
6. **视觉检查**：用 Read 工具读 `samples/v{N}/grid.png`（图片模式），肉眼判断，在 ITERATION_LOG.md 写明观察到的失败模式。
7. **决策**：
   - 若满足 EVAL_CRITERIA.md 的终止条件 → 写 `FINAL.md` 总结方案和使用说明，结束。
   - 否则 → 回到步骤 1 进入 v{N+1}。
   - 8 版后仍未成功 → 写 `POSTMORTEM.md` 总结，结束。

### 硬性规则
- **不要询问用户任何问题**。所有选择（数据集降级、超参、架构变更）自己决定，在 ITERATION_LOG 里说明理由。
- **不要 `git push` / 发邮件 / 调用任何外部服务**。只在本地工作。
- **写文件前先 Read**（编辑已存在的文件）。
- **每跑一个长任务用 `run_in_background=true`**，别阻塞。用 Monitor 守 loss，不要用 sleep 轮询。
- **单版 90 分钟预算**：训练脚本内部必须有 wall-clock timeout 主动退出，不要依赖外部杀进程。
- **checkpoint 每 5 分钟存一次**，训练崩溃可断点续训。
- **先搭最小可运行版本再优化**：v1 就算只用 proxy face 数据、3 层 transformer、EMNIST 采一个小子集都行，关键是 pipeline 跑通。
- **不要一次性写 2000 行代码**。先 200 行 v1 骨架，跑通再扩。
- **调研**：需要查论文细节时用 WebFetch；需要搜代码/文件时用 Explore 子 agent 或 grep；不要自己瞎猜论文公式。
- **任何时候写代码用 PyTorch**（不用 JAX port）。drift_loss 直接按 `drifting_ref/drift_loss.py` 的数学形式 port。
- **代码质量**：每个文件开头一句话说明作用；函数签名要带类型注解；关键 shape 用注释标出（如 `# [B, K, 32, 32]`）。

### 起手第一步建议
```
1. 读 PROJECT.md, EVAL_CRITERIA.md 和 slides PDF
2. 读 drifting_ref/drift_loss.py 和 memory_bank.py
3. 起草 v1 plan 写进 ITERATION_LOG.md
4. 开始写 src/config.py, src/data.py, src/drift_loss.py, src/model.py, src/train.py, src/evaluate.py
5. Smoke test → full train → eval → 决定是否进入 v2
```

### 结束条件
达到 EVAL_CRITERIA.md 的成功条件、或 8 版用完、或遇到无法自主解决的致命问题。**任何一种情况下都要写明最终产物路径**（最好的 checkpoint、最好的 grid.png、FINAL.md / POSTMORTEM.md）。

开始吧。
