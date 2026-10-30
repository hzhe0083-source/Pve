# 拐点数据特化 + RL：选题报告（2026-10-30）

> 这一版取代 [embodied-topic-pick.md](embodied-topic-pick.md)。旧报告推荐的是模型分析题（WAM 噪声诊断），评测要跑很多 rollout，不符合「不打 benchmark、做训练方法」的要求。

## 你要的方向

1. **训练数据特化，围绕「拐点」**：以 [Memory Anchors](https://huggingface.co/papers/2608.26545)（2608.26545，Shuran Song 组）为参照。少量关键数据决定了遗忘：新旧任务在 latent 里重叠、但动作冲突的那些点。论文自动把它们找出来，再把训练预算倾斜过去。
2. **RL**：以 [AlphaNIST](https://huggingface.co/papers/2610.13354)（2610.13354，NVIDIA）为参照。它的做法是 reset 状态库、success-guided sampling（SGS，按成功率把经验集中到能力边界）和稀疏二值奖励。

两者的共同主线是：**自动找出拐点状态，再把训练预算重新分配给它们。**

## 流程

- 两篇锚点论文都读了全文。
- 4 路 agent 分头出题：Memory Anchors 的延伸、AlphaNIST/SGS 的缩小版、两者的桥接、全池撞车扫描。共读约 15 篇全文，产出 22 个想法。
- 合并打分后剩 10 个候选，权重：简单度 0.35、新颖性 0.25、契合度 0.20、故事 0.20。
- 前 3 名做了对抗核查：共 5 次撞车/可行性反驳，外加对首推题的一次独立可行性核查。
- 结论：**3 个都「换个定位后成立」**（holds_if_reframed），没有一个能按原样直接做。下面写的都是调整后的版本。

---

## 首推：RL 在哪里遗忘？拐点 vs 未访问状态（C3 调整版）

> 完整实验设计见 [c3-experiment-design.md](c3-experiment-design.md)（v3，顶会版），本节的设置、预测和时间表以那份文件为准。v3 的主线改为：在预训练 VLA（Evo-1 + RLinf GRPO，LIBERO 同场景任务对）上检验「RL 后训练不忘」在同一场景、新指令时是否成立；PushCube 玩具退为机制实验室。算力约 15–18 GPU-days。

**一句话**：机器人 RL 学新任务时，旧任务会在哪些状态上被遗忘？有两种对立的说法：
- Memory Anchors 说集中在「重叠 + 动作冲突」的拐点；
- [Wolczyk et al. 2024](https://arxiv.org/abs/2402.02868)（FPC）说集中在新任务从不访问的状态。

这个题要做两件事：先正面检验哪种说法对；再看把很小的保持预算（0.5–5%）投到对的位置，能不能比均匀投放更有效。

**为什么首推**：
- 同时用上两篇锚点：Memory Anchors 的拐点检测器，加 AlphaNIST 的 reset 状态库和 SGS（SGS 在这里作为对照组）。
- 最便宜：ManiSkill3 state-based PPO，PushCube 一次训练不到 1 分钟（[PPO README](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo)）。整个主实验约 30 GPU-h，含试点和第二组任务也不超过 5 GPU-days。
- 两种结局都能写：即使「分配」没有效果，「遗忘发生在哪」的定位结果本身就是一个发现。
- 撞车检索确认：**还没人用 Memory Anchors 的判据选 reset 状态来做 RL 中的保持，也没人正面比较过 MA 和 FPC 两种遗忘解释。**

**设置**：
- 环境：ManiSkill3 GPU 仿真，state 观测加 task one-hot，基于 CleanRL 风格的 `examples/baselines/ppo/ppo.py`。
- 任务 A = PushCube，任务 B = PullCube。两者场景和观测完全相同，只有目标在 ±x 两侧，所以需要写一个按 env 翻转目标的子类，约 300–400 行。
- 机器人在开局接近方块时就要走向相反两侧，这里就是天然的「重叠 + 冲突」拐点。
- **训练用 dense reward，二值成功率只作为评测指标。** 核查没有找到稀疏奖励 PPO 能从零学会这些任务的证据。

**做法**：
1. 训练 π_A，从它的 rollout 中快照约 2 万个 env_state 存进 A 状态库。ManiSkill 自带 `reset(options={env_idx, reset_to_env_states})` 和 `CachedResetWrapper`，wrapper 约 50–100 行。
2. 用同一个策略继续训练 B。每 K 轮，在当前 actor 的倒数第二层特征里把 A 状态库分成三块：
   - **ANCHOR**：Memory Anchors 三步法。取落在 A 流形内的 B 状态，按动作漂移取前 10%，再取它们最近邻的 A 条目。
   - **FAR**：离所有 B 状态最远的 A 条目，对应 FPC 说的「未访问」状态。
   - **RANDOM**：随机。
3. **拐点子集指标**：从每块各抽 64 个条目作为起点，测 A 的成功率随训练的变化，成本很低。
4. **干预**：分出 0.5/1/2/5% 的 minibatch 预算做 on-policy 蒸馏。把少量 env reset 到选中的条目，用当前策略往前滚 10–20 步，加一项 λ‖μ_θ(s, A) − μ_{π_A 冻结副本}(s)‖²。教师用冻结的 π_A，因为缓存动作覆盖不到滚动出来的新状态。
5. **对比的分配方式**：anchor、far、uniform、SGS（按 A 的成功率被动分配）、MIR 式漂移。另加一组只用 PPO 奖励、不蒸馏的 anchor-reset，用来检验 RL 通道本身。

**可证伪预测**：
- **P1（定位）**：顺序训练时，A 在 ANCHOR 起点上的成功率下降至少是 RANDOM 的 2 倍，并且大于 FAR。如果 FAR 下降更多，说明 FPC 对、Memory Anchors 的解释不适用于 RL。
- **P2（分配）**：1% 预算下，anchor 蒸馏比 uniform 蒸馏把 A 的最大跌幅降低 ≥40%，比 SGS 降低 ≥25%，并且 B 达到 80% 成功率所需的轮数不变慢。
- **P3（通道）**：只用 PPO 奖励的 anchor-rehearsal 和 SGS 差距在 1 个标准误以内。原因是在二值成功奖励下，A 还没开始失败时 advantage≈0，提供不了预防性梯度。

**第 1 周 go/no-go**：跑 PushCube→PullCube，3 个 seed，三组：不 rehearsal、0.5% uniform 蒸馏、1% uniform 蒸馏。三条都满足才继续：
- (a) A 的成功率在每个 seed 都下降 ≥30 点；
- (b) ANCHOR 和 FAR 的成功率差 ≥15 点，并且 B 训到一半时 ANCHOR 仍然非空；
- (c) 1% uniform 蒸馏后仍有 ≥10 点跌幅，这样才有改进空间。

另外要先验证：重放一条 π_A rollout，结果和原始一致。`set_state_dict` 可能不恢复控制器状态。

- (b) 不过：遗忘是均匀的，没有拐点可选，改题。
- (c) 不过：uniform 已经够用，退化成只做「遗忘定位」的分析短文。

**10 周计划**：

| 周 | 内容 | 检查点 |
|---|---|---|
| W1 | 跑通 ppo.py、写目标翻转子类、验证重放一致性、跑试点 | **Go/No-go** |
| W2–3 | 状态库、三分区、冻结教师蒸馏、rehearsal env | 分区可视化 |
| W4 | P1 定位实验（5 seed） | **核心图：遗忘发生在哪** |
| W5–6 | 分配扫描：6 种分配 × 4 档预算 × 5 seed，约 120 次、约 30 GPU-h | P2 主表 |
| W7 | 第二组同场景任务对（PokeCube 多了一根 peg，不能共用状态库，需要另选） | 泛化 |
| W8 | P3 通道实验，λ 和刷新间隔 K 的消融 | |
| W9–10 | 统计（配对 seed、bootstrap）、出图、写作 | workshop / RA-L 短文 |

**要引用并划清界限的工作**：
- [Memory Anchors 2608.26545](https://arxiv.org/abs/2608.26545)：只做模仿学习，不涉及 RL。
- [Wolczyk 2402.02868](https://arxiv.org/abs/2402.02868)：提出 FPC 解释，本题与它正面对比。
- [ACCRUE 2610.18596](https://arxiv.org/abs/2610.18596)：在 Isaac Lab 装配任务上，对保留 episode 做均匀蒸馏，遗忘已接近 0。这是主要的「改进空间」风险，第 1 周的 (c) 就是在检验它。
- [ReMind 2606.03087](https://arxiv.org/abs/2606.03087) 和 [Continual Prompt Replay 2608.18574](https://arxiv.org/abs/2608.18574)：LLM 领域里按成功率被动 rehearsal 的工作，相当于本题 SGS 组的先例。
- [Kessler 2106.02940](https://arxiv.org/abs/2106.02940)：「同状态不同任务」导致干扰的原始表述。
- [SGS 2610.12465](https://arxiv.org/abs/2610.12465)、[PLR 2010.03934](https://arxiv.org/abs/2010.03934)、[Continual World 2105.10919](https://arxiv.org/abs/2105.10919)。

**主要风险**：最可能的是各种分配方式之间测不出差别。
- 一个原因：任务只随机化方块的二维起点，均匀抽 2 万条目可能已经覆盖了拐点。
- 另一个原因：FAR 条目往往是「方块已在目标位置」的末期状态，天然就算成功。所以要先剔除已成功的条目。
- 定位实验 P1 不依赖分配是否有效，这是兜底。

---

## 备选：拐点重加权修的是「选错」，不修「做不准」（C1 调整版）

这是最贴近 Memory Anchors 的纯数据特化题，在普通的多任务模仿学习里做，不涉及持续学习。

**问题**：LIBERO-Goal 的 10 个任务共用同一个场景，策略常犯「选错物体或目标」的错。把跨任务拐点帧（重叠 + 冲突）在 batch 里过采样，能不能减少这类错？核查后的预期是：**只能修「选错」，修不了「做不准」，并且只在数据少时有效。** Memory Anchors 附录 C.1 指出，数据量大时这些锚点本来就会被均匀采到。

**设置**：
- 策略：语言条件的 Diffusion Policy（ResNet-18，16 步 chunk）。可以用 robomimic v0.5 的 DP 加 CLIP `lang_emb`；LeRobot 的 diffusion 不带语言条件，需要自己加。
- 数据：LIBERO-Goal，每任务 10、25、50 条 demo 三档。
- 检测器：Memory Anchors 的小噪声去噪判据搬到联合训练里会失效，因为兄弟任务的模式都在训练分布内。改用「留出 probe 帧的分支错误率」：从纯噪声采 K=8 个 chunk，如果超过 25% 更接近兄弟任务的真值，就标记为拐点。约 200 行。

**对照组**（ρ=0.2，同样步数和数据）：
- uniform；
- Memory Anchors 拐点；
- **同等大小的时间/夹爪事件启发式**（前 15% 帧加抓取后的帧）；
- Wen-2021 changepoint 加权；
- RHO-loss 式按 loss 优先；
- 只加权「做不准」的帧；
- 随机子集。

**主指标**：wrong-commit 率。日志器（约 80 行）检查两件事：兄弟任务的 BDDL 目标谓词是否被满足；第一个被移动的物体是否不在本任务的 obj_of_interest 里。

**成本**：12–14 GPU-days，8–10 周。在 31 GB 内存的机器上，并行评测环境要限制在约 8 个以内。

**第 1 周门槛**（全部满足才继续）：
- uniform 基线成功率在 55–85%；
- wrong-commit 至少占 8% 的 episode，且至少占 30% 的失败；
- 检测器标记 2–15% 的帧；
- 标记集合和时间启发式的 Jaccard < 0.7。如果更高，检测器等于启发式，就放弃「检测器」这条主张。

**撞车情况**：「给关键帧加权」这件事已经很拥挤，下面这些都要作为 baseline 或在文中讨论：
- [Keyframe-focused IL 2106.06452](https://arxiv.org/abs/2106.06452)
- [SCRIM 2610.17709](https://arxiv.org/abs/2610.17709)
- [FrameSkip 2605.13757](https://arxiv.org/abs/2605.13757)
- [语言关键步 2606.11906](https://arxiv.org/abs/2606.11906)：同样用了 LIBERO-Goal
- [Balanced BC 2508.06319](https://arxiv.org/abs/2508.06319)
- [STAIR 2606.15587](https://arxiv.org/abs/2606.15587)

「用跨任务的重叠 + 冲突作为 BC 采样权重」这个具体做法，检索中还没有找到先例。

**注意**：Memory Anchors 的代码**没有公开**（项目页写着 "coming soon"），需要按附录复现，约 150 行。可以第 1 天就给一作发邮件要早期代码。

---

## 其他 RL 方向（未经核查，可作为后续扩展）

- **C4 PNR-Bank**：在失败的 rollout 里，用 critic 骤降加仿真分支验证，找到「不可挽回点」，把它前面的状态动态加入 SGS 状态库。这是最直接延续 AlphaNIST 的方向，ManiSkill PegInsertion/PlugCharger 上约 6–8 GPU-days。和 RICE（ICML 2024）、TD-error restart 有部分撞车。
- **C5 兄弟组 reset**：在多任务 PPO 里，把同一个冲突状态同时用各兄弟任务的 ID 启动，让同一个 batch 里出现「同状态、相反 advantage」的对比。这是最贴合两篇锚点的融合，但需要自己搭一个多目标环境，约 4–5 GPU-days。

## 不推荐的

- **C2「Memory Anchors 是数据属性还是表征属性」**：[PHASER 2606.03598](https://arxiv.org/abs/2606.03598) 已经做了不依赖策略的「重叠 × 动作分歧」回放打分。而且效应只集中在少数任务对上，3 个 seed 区分不了 70% 和 30% 的恢复率。
- **C6 Fork-gated DPPO**：拐点是任务内部的多模态，不是跨任务冲突；而且 DPPO 在 robomimic 上已接近饱和。
- **C7**（LIBERO-10 阶段混叠）：可以并入 C1 作为第二个设置。
- **C8、C9**：改动太小，可能被噪声淹没。
- **C10**：SmolVLA 在同场景任务上可能几乎不遗忘，没有改进空间。

## 全部 10 个候选的打分

| cid | 总分 | 简单 | 新颖 | 契合 | 故事 | 题目 |
|---|---|---|---|---|---|---|
| C1 | 7.70 | 9 | 5 | 9 | 7.5 | 多任务 BC 中的拐点过采样 |
| C2 | 7.22 | 7 | 5.5 | 9 | 8 | 不依赖策略的 anchors 与遗忘预测 |
| C3 | 7.20 | 7.5 | 5.5 | 9 | 7 | anchor rehearsal resets：主动 vs 被动经验分配 |
| C4 | 7.12 | 7 | 5.5 | 8.5 | 8 | PNR-Bank：用不可挽回点扩充 SGS 状态库 |
| C5 | 7.12 | 6 | 6.5 | 9.5 | 7.5 | 兄弟组 reset：冲突状态做对比式启动 |
| C6 | 7.12 | 8 | 6.5 | 6.5 | 7 | Fork-gated DPPO |
| C7 | 7.05 | 8 | 5 | 8 | 7 | 长时程任务内的阶段混叠拐点 |
| C8 | 6.70 | 7 | 5 | 8 | 7 | 模仿学习拐点能否预测 RL 拐点 |
| C9 | 6.67 | 9 | 4.5 | 6 | 6 | Pooled-SGS：小规模环境下的 SGS |
| C10 | 6.55 | 5.5 | 6.5 | 8 | 7 | 无原始数据时保护 VLA 技能 |

C3 原始分低于 C1，但核查后排序反过来了。C1 有被「时间启发式等价」和「数据多时无效」两种方式证伪的风险，新颖性空间也更挤。C3 算力最省，第 1 周就能判断能不能做，而且最差也能落到「遗忘定位」这个结论上。
