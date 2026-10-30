# 具身选题：从 HF Daily Papers（2026-09-15 → 10-30）里挑出来的简单可做题

> **已被 [turning-point-topic-pick.md](turning-point-topic-pick.md) 取代。** 这一版的推荐偏模型分析，评测要跑大量 rollout，不符合「训练方法、不打 benchmark」的方向。

**流程**：抓取 46 天的 daily papers，共 1852 篇。关键词筛出 319 篇具身论文，分 8 个子方向各派一个 agent 精读，提出 24 个选题，合并成 14 个候选。之后从可行性、新颖性、故事三个角度打分，再对前 5 名各做一次撞车检索（每题 ≥8 次搜索）和一次可行性核查（clone 仓库、核对权重和显存）。

**约束假设**：共享 4×4090 + 1×5090，总预算不超过 60 GPU-days；不用真机，只用仿真；单人 8–12 周完成，目标 workshop 或 RA-L 级别。

**结论先说**：前 5 名全部被判 *partially_scooped*。这个领域六周内的热点基本都有人碰过，所以下面列出的都是已经按撞车检索调整过定位的版本。

---

## 推荐：C04「The Hidden Seed」：单步纯噪声 WAM 里，世界侧噪声到底在干什么

**灵感来源**：Simple-WAM（[2609.34981](https://huggingface.co/papers/2609.34981)，141 赞）。它只做一次 forward，未来 video token 全部是纯噪声，就在 LIBERO-Plus 上拿到 79.5。

**已核实的代码事实**：
- `src/simplewam/models/wan22/fastwam_joint.py` 第 165–166 行：每次调用 `infer_action` 时，video 和 action 两个 generator 都用同一个 `seed` 重新 `manual_seed`。
- 评测配置继承 `seed: 42`。
- 结果是整个评测里每个 chunk 用的都是同一份未来噪声 ε_v。

**问题**：
1. ε_v 是「随机想象」「确定性寄存器」，还是「噪声即增强」（mimic-video 附录 E 的解释）？
2. ε_v 的 seed 方差，会不会大到盖过近期 WAM 之间 1–3 点的差距？对比的几个数字：Simple-WAM 79.5、RegisterWAM 81.6（其 baseline 76.3）、Faster-WAM 73.6。

**做法**：只改推理，不训练。
1. 把两个 generator 拆开，固定 ε_a、扫 8 个 ε_v seed，再反过来固定 ε_v 扫 ε_a。
2. 每个 chunk 重新采样噪声（fresh noise）。
3. FNM：N 个 ε_v 组成一个 batch 做 prefill，action expert 每步对 N 个 velocity 取平均。K/V 平均和 action medoid 作为消融。
4. 诊断：逐层统计 ε 敏感度；做 D_v / D_a 分解；用 D_v 预测失败，算 AUROC，对照 D_a、VFD 和 accel proxy。

**成本**：约 12–15 GPU-days，纯推理。
- 4090 上默认加载路径放不下：DiT、T5 和 VAE 合计约 25 GB。haha-b/c 只有 31 GB 内存，而加载峰值约 37 GB。需要改约 100–150 行代码：预先缓存 T5 context、设 `load_text_encoder=false`、`torch.load(mmap=True)`。
- 5090 直接能跑。

**证伪与 go/no-go**：
- 第 2 周离线测 D_v/D_a，约 1 GPU-h。
  - 若 < 0.05：放弃方法线，写诊断短文。主张是 ε_v 被当成寄存器用，以及 WAM 评测必须报告噪声 seed。
  - 若 ≥ 0.1：继续做 FNM。
- FNM 有效的判据：3 个 seed 汇总的配对增益 ≥ 1.5 点，且 McNemar p < 0.05。

**统计陷阱**：1,506 个 episode 的 binomial SE 约 1 点，8 个 seed 的极差单靠抽样就能到 1.6–2.3 点。因此必须：
- 用 episode 级的结果翻转率、permutation 检验和配对 McNemar；
- 先用同一 seed 跑两次，测出非确定性带来的翻转下限（torch.compile、cudagraph 和 LIBERO-Plus 的全局 RNG 都可能引入非确定性）；
- 每个 episode 开始时重新 seed。

**撞车情况**：
- FNM 的方法本身约等于 DIFT（[2306.03881](https://arxiv.org/abs/2306.03881)）的噪声集成，必须引用，并降为次要贡献。
- RIFT（[2608.11521](https://arxiv.org/abs/2608.11521)）对 future cache 做过分布外干预，本题是分布内地改输入 ε，两者互补。
- 没有找到 ε_v seed 扫描或 D_v 分解的先例。
- 时间窗口风险高：FRED（[2610.30320](https://arxiv.org/abs/2610.30320)）已经把 h=f(ε|o,c) 写进公式，补做 seed 分析很容易。建议 2–3 周内先把 seed lottery 跑出来，挂 arXiv。

**栈**：
- 代码：[Simple-WAM](https://github.com/LeapLabTHU/Simple-WAM)
- 权重：[rpzhou/Simple-WAM](https://huggingface.co/rpzhou/Simple-WAM)，含 LIBERO full、fewshot5/10、taskgen 和 RoboTwin，每个约 12 GB
- 评测：LIBERO-Plus，使用 [JAM 的 1,506-task 固定子集](https://github.com/y0urOy/JAM/blob/main/reproduction/libero_plus_paper15_tasks.txt)。`run_libero_manager.py` 会覆盖 task_file，需要打约 3 行补丁。
- RoboTwin 太贵，每个配置约 10k episode，不做。

**10 周计划**：
| 周 | 内容 | 检查点 |
|---|---|---|
| W1 | 配环境，写 T5 缓存、mmap、task_file 补丁和每 episode 重新 seed；跑 50 个 episode 冒烟，记录 ε hash，确认每个 chunk 相同 | 4090 单 worker 显存 ≤ 16 GB |
| W2 | 默认 seed 跑两遍 1,506（复现 + 非确定性下限）；离线测 D_v/D_a 和逐层敏感度 | **Go/No-go #1** |
| W3 | 固定 ε_a，扫 8 个 ε_v seed（约 50 GPU-h） | seed lottery 初表 |
| W4 | 固定 ε_v，扫 4 个 ε_a seed；fresh noise ×3；方差分解和翻转率 | Finding 1 定稿 |
| W5 | 实现 FNM（以速度平均为主），在困难类别（camera/noise/background）上做 300 个 episode 的 pilot | **Go/No-go #2**（配对增益 ≥ 1 点） |
| W6 | FNM N=8 ×3 seed 全量，N=4 ×1 | 主表 |
| W7 | 加 fewshot10 checkpoint 作为第二模型；在时间对齐窗口内做失败预测 AUROC | **Go/No-go #3**：定论文形态 |
| W8 | 消融（层位置、聚合方式）；预算允许时跑 10,030 全量，对照 79.5 | |
| W9 | 统计复核和出图 | |
| W10 | 写作，投 NeurIPS/ICLR workshop；若 FNM 增益 ≥ 2–3 点且显著，改投 RA-L/ICRA | |

---

## 备选 1（最省，不需要仿真器）：C12「PRM 的 context 缺口在哪」

**灵感来源**：ProgressCompass（[2609.36684](https://arxiv.org/abs/2609.36684)）。它用 27B + 9B 的 VLM 循环把冻结 PRM 的 MAE 从 25.4 降到 9.3，但每个 episode 要 39.5 s。

**做法**：
- 在 ContextProgress-Bench 上做因子分解，benchmark 只有 120 段视频，不用跑仿真。
- 三个因子：plan（oracle / VLM 单次）× boundary gate（Robometer success 头 / progress 头 / PC 式 proposer / Qwen 验证 / oracle）× 组合方式（k/K 计数 / self-chained）。
- 方法贡献：selective verification。只有 success 概率落在不确定带时才调用小 VLM，报告 VLM 调用率和 MAE 的 Pareto 曲线。

**成本**：不超过 5 GPU-days，一张 4090 就能起步。

**撞车**：Robometer 原文（[2603.02115](https://arxiv.org/abs/2603.02115) App. E-2）已经用 success > 0.6 推进 stage；ProgressCompass 的 τ 消融也已回答了一半问题。所以只能定位成诊断加强基线，不能声称是新方法。

**第 1 周 go/no-go**：复现 raw RoboMeter MAE 到 25.4±2.5 以内，oracle-context MAE 落在 3.4–6.7。

**栈**：
- 数据：[ContextProgress-Bench](https://huggingface.co/datasets/Sterzhang/ContextProgress-Bench)
- 模型：[Robometer-4B](https://huggingface.co/robometer/Robometer-4B)（Apache-2.0）

## 备选 2（新颖性较高，但更重）：C13「记忆型 VLA 数的是尝试还是结果」

**做法**：
- 在 RoboMME 的计数任务上，对已发布的 learned memory π0.5 变体（14 个目录里实际只有 10 个有权重）做评测期反事实。
- 两类扰动：a1 drop-after-lift、a2 undo-at-release，另加一组 sham 对照。
- 主指标是带符号计数误差。

**撞车**：AGM（[2608.29537](https://arxiv.org/abs/2608.29537)）已经提出「attempt ≠ achievement」，但只针对显式记忆。EventSync（[2610.27917](https://arxiv.org/abs/2610.27917)）是 10 天前的工作，也在附近。本题的新意落在 learned 隐式记忆，以及双向反事实设计。

**成本与风险**：约 12–16 GPU-days。环境是 JAX/openpi 加 ManiSkill，jax 0.5.3 在 5090 上能不能跑还需要实测。

---

## 不推荐的（及原因）

- **C05 Foresight vs Hindsight（JAM）**：在 DINOv2 平台上，foresight 的总效应只有约 2 点，落在 seed 噪声以内；NowWAM 已经报告 past≈future。
- **C11 LightNav-0 `<opos>` 审计**：StopWatch 已经量化过 LightNav-0「到了不停」；而且 stop 哨兵在 `<apos>`（1299），不在 `<opos>`。
- **C01 RL schedule-lock**：第二阶段需要在消费级卡上跑 π0.5 PPO，不现实。第一阶段的零训练诊断（用 RLinf 公开 ckpt 换去噪步数）可以单独做，算一个小备选。
- **C07 World-CFG**：需要 Isaac/RoboDojo，环境太重。

## 全部 14 个候选的打分（加权：可行性 0.45、新颖性 0.30、故事 0.25）

| cid | 总分 | 可行 | 新颖 | 故事 | 题目 |
|---|---|---|---|---|---|
| C05 | 6.98 | 7.5 | 5.5 | 7.8 | Foresight 还是 Hindsight：拆解 VLA 未来特征对齐 |
| C04 | 6.80 | 8.0 | 5.0 | 6.8 | 固定噪声即寄存器：Simple-WAM 纯噪声未来 + 噪声边缘化 |
| C12 | 6.80 | 8.5 | 4.5 | 6.5 | 冻结 PRM success 头 + 计数器替代 VLM 编排 |
| C13 | 6.70 | 6.5 | 6.5 | 7.3 | 记忆型 VLA 计数的 retry/phantom 反事实诊断 |
| C11 | 6.63 | 7.5 | 5.0 | 7.0 | LightNav-0 物体指向通道的因果审计 |
| C01 | 6.50 | 5.5 | 6.5 | 8.3 | RL 增益是否锁死在去噪时间网格上 |
| C07 | 6.25 | 5.0 | 7.5 | 7.0 | World-CFG：WAM 世界分支作推理期 guidance |
| C06 | 6.23 | 5.0 | 7.0 | 7.5 | 无动作视频教会了 WAM 什么 |
| C09 | 6.00 | 7.0 | 5.5 | 4.8 | Fork-gated Flow：少步去噪只在岔路口失效？ |
| C14 | 5.88 | 6.0 | 6.0 | 5.5 | 可挽救窗口预测快慢 VLA 切换收益 |
| C02 | 5.50 | 6.0 | 4.5 | 5.8 | FERT 检索 vs 阶段风险加权 |
| C10 | 5.43 | 6.0 | 4.5 | 5.5 | Warm-start 让 flow VLA 反应迟钝吗 |
| C03 | 5.00 | 5.0 | 5.0 | 5.0 | 执行误差补偿压进时间步调制通路 |
| C08 | 4.50 | 5.5 | 3.0 | 4.5 | WAM 加未来接触预见 |

C04 排在 C05 之后，但经过撞车和可行性复核后，C05 暴露出效应量太小的问题，所以最终推荐换成 C04。
