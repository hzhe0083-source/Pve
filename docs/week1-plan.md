# 第 1 周执行计划：Where to Reuse, Where to Learn

> 方法的来龙去脉见 [final-topic-report.md](../final-topic-report.md)，第 11 节是「越学越快」核查后的修正。本文以数学框架为主线：先给出全部公式，再写每条公式落在哪段代码、哪一周实现、怎么验证，最后是第 1 周的清单、闸门和已有结果。

## 0. 已定的决策

1. **主线**：小策略加大规模仿真 RL，和 AlphaNIST、组里的 E-10 同一类。平台是 ManiSkill3 的 Panda 任务，先用 state 观测。VLA 只有在 G1、G2 都通过、并且还有余力时才做。
2. **方法**：先单独做 Where to Reuse, Where to Learn。Aliased, Not Overloaded 作为「并入库」那一步吸收进来（F9），第 4 周以后再做。
3. **任务池**：8 个任务，即 PushCube、PullCube、PickCube、StackCube、PlaceSphere、PokeCube、LiftPegUpright、PullCubeTool；另加远端目标 PegInsertionSide。
4. **机器**：全部在 5090d 上跑，用你的 zhr 账号。代码在本地 `~/Documents/pve`（分支 `wrwl/week1`），用 rsync 同步到 `5090d:~/pve`。不碰 a2r 仓库和组里的环境。
5. **GPU 使用**：每次启动前先跑 `gpumon free`，跑完清理自己的进程。组里的 E-10 也用这台机器，被占时就等。
6. **组里的约束**：关键度只用来分配更新和决定照抄，**永远不用来挑重置起点**。重置和起点采样是组里的工作。

---

## 1. 数学框架

### 1.1 记号

- 一个**决策**是一个动作块，持续 $\delta$ 秒，记为 $t = 0, \dots, H-1$。成功 $Y \in \{0,1\}$ 在回合结束时判定一次。
- $Q^\pi_t(s,a) = P(Y=1 \mid s_t=s,\ a_t=a,\ \text{之后按 } \pi)$；$V^\pi(s) = \mathbb{E}_{a\sim\pi}\,Q^\pi(s,a)$；$J(\pi) = P_\pi(Y=1)$。
- $\pi_{\text{base}}$：新任务从零起步时的初始化策略。
- 温度为 $\tau$ 的**熟练策略**：

$$\pi^*_\tau(a\mid s) \;\propto\; \pi_{\text{base}}(a\mid s)\; e^{\,Q^{\pi^*}(s,a)/\tau}$$

- **库** $L = \{\pi_1,\dots,\pi_K\}$ 是冻结的已学会任务的策略；$\phi$ 是共享主干，它的输出就是「潜空间」。

### 1.2 两种关键度：一个给照抄定价，一个给学习定价

**(F1) 照抄的代价权重：关键度**

$$c^*(s) \;=\; \mathrm{Var}_{a\sim\pi^*_\tau}\big[\,Q^{\pi^*}(s,a)\,\big]$$

**(F2) 要学多少：积分关键度**

$$\bar c(s) \;=\; 2\tau^2\,\mathrm{KL}\big(\pi^*_\tau \,\|\, \pi_{\text{base}}\big)(s) \;=\; 2\int_0^1 u\,\mathrm{Var}_{\pi_{u/\tau}}[Q]\,du$$

两者的关系是 $e^{-R/\tau}\,c^* \le \bar c \le e^{R/\tau}\,c^*$，其中 $R$ 是 $Q(s,\cdot)$ 的取值范围。

**为什么需要两个**：熟练策略在关键的离散选择上执行得很稳定时，$c^*$ 会趋近 0，比如推还是拉、从哪一侧接触。但这一步要学的信息量 $\bar c$ 仍然很大。所以 $c^*$ 只用来衡量照抄的代价，$\bar c$ 才用来衡量还要学多少。

**估计方法（第 2 周实现，一组样本同时给出两个量）**：
1. 把仿真恢复到状态 $s$，从混合提议分布 $\mu = \tfrac12\pi_{\text{task}} + \tfrac12\pi_{\text{base}}$ 采 $M$ 个第一步动作块 $a_m$。
2. 每个动作块接着用 $\pi_{\text{task}}$ 做完 $N$ 次，得到成功率 $\hat q_m$。
3. 计算权重：

$$w_m = \frac{\pi_{\text{base}}(a_m)}{\mu(a_m)}, \qquad \tilde w_m \propto w_m\, e^{\hat q_m/\tau}, \qquad \hat Z = \frac{\sum_m w_m e^{\hat q_m/\tau}}{\sum_m w_m}$$

4. 得到两个估计量：

$$\widehat{\mathrm{KL}}(s) = \sum_m \tilde w_m \frac{\hat q_m}{\tau} - \log \hat Z, \qquad \hat c^*(s) = \sum_m \tilde w_m \hat q_m^2 - \Big(\sum_m \tilde w_m \hat q_m\Big)^2 - \sum_m \tilde w_m \frac{\hat q_m (1-\hat q_m)}{N-1}$$

最后一项扣除的是二项分布的噪声。$\tau$ 只按分半一致性选一次，之后预注册、不再改。

### 1.3 理论依据

**(F3) 方差预算**：在确定性仿真里取等号。

$$\sum_t \mathbb{E}_{d^\pi_t}\, c^\pi(s_t) \;\le\; J(\pi)\big(1 - J(\pi)\big)$$

它的含义是：SGS 的 $p(1-p)$ 告诉你哪些回合带信号，$c$ 告诉你回合里哪一步带信号。

**(F4) 共享代价引理**：用别的策略代替自己的策略，代价集中在关键且冲突的决策上。

$$J(\pi^*) - J(\pi) \;\le\; \sum_t \mathbb{E}_{s\sim d^\pi_t}\sqrt{\chi^2(\pi\,\|\,\pi^*)(s)\; c^*(s)}$$

**(F5) 屏蔽引理**：只在决策集合 $M$ 上照抄库的先验 $p_L$，其余由熟练策略来做（记为 $\tilde\pi_L$）。

$$J^* - J(\tilde\pi_L) \;\le\; \sqrt{\mathbb{E}\sum_{t\in M}\chi^2(p_L\,\|\,\pi^*)}\;\cdot\;\sqrt{\mathbb{E}\sum_{t\in M} c^*}$$

取 $M$ 为 $c^*$ 质量最低的 $(1-\epsilon)$ 那部分时，代价为 $O(\sqrt{\epsilon}\,\sigma^*)$。

### 1.4 库先验和门控：哪里照抄、哪里自学

**(F6) 库先验、熟悉度和一致度**

$$p_L(\cdot\mid s) = \sum_j w_j(s)\,\pi_j(\cdot\mid s), \qquad w_j(s) \propto e^{-d_j(s)^2/\tau_d^2}, \qquad \kappa(s) = \max_j e^{-d_j(s)^2/\tau_d^2}$$

- $d_j(s)$：在 $\phi$ 的潜空间里，$s$ 到任务 $j$ 访问过的状态的近邻距离。
- **一致度** $A(s) = \mathrm{JS}\big(\pi_{j_1}, \pi_{j_2}, \pi_{j_3}\big)(s)$，衡量检索到的前 3 个源策略之间的分歧。
- **防御性先验** $\tilde p_L = \tfrac12 p_L + \tfrac12\pi_{\text{base}}$：每个决策最多多付 $\log 2$ nats，但能避免被冲突的源带偏。

**(F7) 门控**

- **冷启动**，新任务还几乎不成功，只能靠预测：

$$g(s) = \mathbb{1}\big[\hat\rho^*(s) \le \rho_\epsilon\big]\cdot \mathbb{1}\big[A(s) \le A_{90}\big]\cdot \mathbb{1}\big[\kappa(s) \ge \kappa_0\big], \qquad \hat\rho^*(s) = g_\psi\big(\phi(s)\big) \approx \frac{c^*(s)}{\mathbb{E}\sum_t c^*(s_t)}$$

  $\rho_\epsilon$ 的取法是：被照抄的决策合计只占预测 $c^*$ 质量的 $\epsilon$；$A_{90}$ 是库内一致度的 90 分位。
- **有了成功信号之后**（$\hat\sigma_k > \sigma_{\min}$ 时切换）：

$$g(s) = \sigma\!\left(-\frac{\Delta_0(s)}{s_0}\right), \qquad \Delta_0(s) = V(s) - \mathbb{E}_{a\sim\tilde p_L}\,Q(s,a)$$

- **理想门控**，只用于分析：当且仅当 $\mathrm{KL}(\pi^*\,\|\,p_L)(s) < \mathrm{KL}(\pi^*\,\|\,\pi_{\text{base}})(s)$ 时照抄，也就是库的先验比从零开始更接近熟练策略。

### 1.5 更新规则

**(F8) 协变的 PPO 式目标**，按决策（动作块）计算比率 $r_s = \pi_\theta(a_s\mid s)/\pi_{\text{old}}(a_s\mid s)$：

$$\mathcal{L}(\theta) = \mathbb{E}_s\!\left[\, r_s\,\frac{\hat A_s}{\hat\sigma_k} - \frac{(r_s-1)^2}{2\sqrt{2\Delta}} \,\right] \;-\; \lambda_0\,\mathbb{E}_s\!\left[\, g(s)\,\mathrm{KL}\big(\pi_\theta \,\|\, \tilde p_L\big)(s) \,\right]$$

- 第一项负责学习：每个决策的最优解是 $r_s - 1 = \sqrt{2\Delta}\,\hat A_s/\hat\sigma_k$，所以 $\chi^2_s \approx 2\Delta\, c(s)/\sigma_k^2$，**更新量自动正比于关键度**，每回合的 KL 约等于 $\Delta$。
- 第二项负责复用：只在 $g = 1$ 的地方向库靠拢。用反向 KL，$\lambda_0$ 随训练退火。
- 按任务归一化：$\hat\sigma_k^2 = \max\big(\widehat{\mathrm{Var}}_k(Y),\ \sigma_{\min}^2\big)$。
- GAE 按物理时间设置：$\gamma = e^{-\delta/T_\gamma}$，$\lambda = e^{-\delta/T_\lambda}$。
- 超参数只有 $\Delta$、$\epsilon$、$\lambda_0$、$\sigma_{\min}$，以及两个时间常数。

### 1.6 并入库

**(F9) 只在关键的混叠处分离表示**：$\mathcal{A}$ 是混叠且动作冲突的状态对，即潜空间距离小于 $r$，并且两任务的动作分歧 JS 超过阈值。

$$\mathcal{L}_{\text{sep}} = \sum_{(s_i,s_j)\in\mathcal{A}} \big(c^*_{\tau,i}(s_i) + c^*_{\tau,j}(s_j)\big)\cdot \max\!\big(0,\ m - \|\phi(s_i,i) - \phi(s_j,j)\|\big)^2$$

这里的关键度用带温度的策略计算，所以在执行稳定的离散选择上也不会消失。总损失再加上把每个任务蒸馏进共享网络的那一项。

### 1.7 学习时间规律

**(F10) 剩余信息量**：$U_L$ 是库覆盖不了的决策，即 $g = 0$ 的地方。

$$D(t\mid L) = \sum_t \mathbb{E}_{d_t^{\tilde\pi_L}}\!\Big[\mathbb{1}_{U_L}(s_t)\,\mathrm{KL}\big(\pi^*_\tau\,\|\,\pi_{\text{base}}\big)(s_t)\Big], \qquad \nu(t\mid L) = \frac{D(t\mid L)}{D(t\mid\varnothing)}$$

**经验规律（待检验）**：$T_{80}(t\mid L) \approx a + b\,\nu(t\mid L)$。

- 能证明的部分：对理想化的学习器，学习时间有一个随 $D$ 线性增长的上界；新内容是链式结构时这个界是紧的；对任务顺序取期望时，在考虑冲突的门控下，这个界随库变大单调、凸地下降。
- 对 PPO 本身，只叫「经验规律」，不叫定理。

### 1.8 哪些是新的

| 新的（论文的贡献） | 借用的（必须引用） |
|---|---|
| D1：$c^*$ 和 $\bar c$ 能从冻结的潜空间零样本预测 | $c^*$ 的定义（Karino 2020、Li 2026、Russo 2026） |
| F4、F5 用于「哪里复用」；F7 门控（关键度 × 一致度 × 熟悉度） | F8 的协变更新（NPG、V-MPO、PPO-λ） |
| F10：$\nu$ 预测学习时间，并随库变大递减 | KL 先验（Distral、kickstarting、ASPiRe） |
| F9 按关键度加权的局部分离 | 「重叠 × 冲突」的干扰（Memory Anchors） |

---

## 2. 公式 → 代码 → 周次

| 公式 | 代码 | 周 | 怎么验证 |
|---|---|---|---|
| 统一观测（所有公式的前提） | `wrwl/envs/unified.py` | W1 | ✅ 9 个任务都能跑，78 维 |
| F8 协变更新（先令 $\lambda_0 = 0$）、$\hat\sigma_k$、物理时间 GAE | `wrwl/ppo.py` | W1 | G1 第 1、4 条 |
| 状态恢复（F1、F2 估计的前提） | `scripts/restore_check.py` | W1 | G1 第 2 条，要用熟练策略测 |
| 从零训练 vs 直接微调（F4、F5 有用的前提） | `scripts/e0.py` | W1 | G1 第 3 条 |
| F1、F2 的估计（混合提议分布的分支 rollout） | `wrwl/labels.py` | W2 | 分半一致性 ≥ 0.6；用 F3 做合理性检查 |
| D1 探针 $g_\psi$（预测 $c^*$ 和 $\bar c$ 的份额） | `wrwl/probe.py` | W2 | G2 |
| F4、F5 的数值检验 | `analysis/` | W2–3 | 实测亏损 vs 理论上界 |
| F6、F7 先验和门控 | `wrwl/prior.py` | W3 | G3 |
| F10 的 $\nu$ | `analysis/` | W2（闸门 0）、W5–8 | 闸门 0、1 |
| F9 并入库 | `wrwl/consolidate.py` | W4 以后 | 闸门 1 |

---

## 3. 第 1 周清单

| 天 | 工作 | 状态 |
|---|---|---|
| 1 | 5090 环境、冒烟测试、状态恢复（随机动作） | ✅ 完成 |
| 1–2 | 统一观测的多任务环境 | ✅ 完成，速度待优化 |
| 2–3 | `wrwl/ppo.py`：在 ManiSkill 的 ppo 基础上实现 F8。按决策算比率（动作块长度 H 可选），按任务的 $\hat\sigma_k$ 带下限，每回合 KL 预算 $\Delta$，物理时间的 $\gamma$ 和 $\lambda$；同时保留 PPO-clip 开关，作为对照 | ✅ 代码完成，已冒烟测试：`--loss cov`、每轮 KL 记到 `train.csv`、`--control_freq` 和 `--max_episode_steps`、统一观测模式 `--unified`。KL 缩放实验已排队 |
| 3–4 | E0：源库（PushCube、PullCube、PickCube、StackCube、PokeCube）的多任务策略，3 个 seed；新任务 PlaceSphere、PullCubeTool、LiftPegUpright 上比较从零训练和直接微调，各 3 个 seed。PegInsertionSide 从零训练太慢，换成 LiftPegUpright；源库蒸馏时排除新任务 | 已排队（`scripts/run_week1_rest.sh`） |
| 4 | 用熟练策略重测状态恢复：结局能否复现，每个状态重复 K=8 次，噪声和原轨迹相同（`scripts/restore_check_expert.py`） | 已排队。冒烟测试（PushCube、PickCube、PullCube，每个 64 个状态）结局一致率 0.97；PickCube 的副本之间几乎完全一致，但和原轨迹差几毫米，说明恢复丢了一点接触状态，而不是仿真本身不确定 |
| 4–5 | **手工门控**：在 PullCubeTool 和 PlaceSphere 上，人为指定接近、抓取这两个阶段照抄库，钩住、放置等阶段自学，F8 里的 $g$ 按阶段手工设定；和直接微调、从零训练对比，3 个 seed。它是自动门控（F7）效果的上界，用来提前检验「按决策复用能消掉负迁移、保住正迁移」这个原理，约 1–2 GPU-h。实现：新任务的 one-hot 列复制指定源任务的权重（PullCubeTool ← PokeCube，PlaceSphere ← PickCube），先验是初始网络的冻结副本（`--prior_self`），没抓住东西时 $g=1$、抓住后 $g=0$（`--gate pregrasp`）；另加处处 KL 的 kickstart 对照 | 已排队，和 E0 共用从零、微调两组 |
| 5 | 检查 G1（`scripts/analyze_g1.py`，队列最后自动跑，输出 `logs/g1.txt`、`figures/g1_kl_scaling.png`、`figures/e0_handgate_curves.png`） | 已排队 |

---

### 第 1 晚根据评审做的修正（2026-10-30 23:30）

- **恢复后抓取标志读到旧值**：ManiSkill 恢复状态后的第一步，`is_grasping` 读的是这个环境槽位上一次仿真的接触力。标签和恢复检查现在改用采集时缓存的观测和专家动作做第一步。修正后，PickCube 恢复的结局一致率从 0.95 升到 1.00，位置误差中位数从 5.8 mm 降到 1.0 mm。
- **标签预算**：分支从 8×4 改成 16×8（每个状态 128 次续跑），每个任务 256 个状态，并输出分半一致性。小样本冒烟测试：分半 Spearman 0.72–0.88。
- **StackCube 专家**：25M 步只到 0.69，低于 G1 第 1 条的 0.8。从最终检查点续训 15M 步；如果更好就替换，原检查点留作 `final_ckpt_25M.pt`。隐空间流水线会等续训结束再开始。
- 评审认为今晚的 M1 和 E0 只能当预实验：稠密奖励、手工门控、平移过的起始区域、只有 3 个 seed。能写进论文的版本要按评审意见重做。

## 4. 闸门 G1（四条都要满足）

1. 每个源任务成功率 ≥ 80%。
2. 用熟练策略测，恢复状态后 ≥ 95% 的重放能复现结局。这是 F1、F2 估计的前提。
3. 迁移两个方向都有空间：直接微调在某个新任务上比从零快 ≥ 1.3 倍，在另一个新任务上不比从零好。这是 F4、F5 有意义的前提。
4. F8 的性质成立：
   - 协变版每回合的 KL 约等于 $\Delta$，和决策拆分数 $m$ 无关，波动在 ±20% 以内；
   - PPO-clip 每回合的 KL 随 $m$ 翻倍而增长 ≥ 1.8 倍（截断把每个决策的 KL 限在差不多同一个量级，所以每回合的 KL 和 $m$ 成正比）；
   - 协变版最终成功率和调好的 PPO-clip 差距在 10% 以内。

**附加证据，不作为闸门**：手工门控在冲突任务上比直接微调快，在一致任务上不比直接微调慢。这说明原理成立；自动门控能否做到同样的效果，要等第 3 周的 G3。

**不过怎么办**：
- 第 1、2 条不过：先修好。
- 第 3 条不过：换新任务，比如 LiftPegUpright。
- 第 4 条的性能部分不过：底座退回 adaptive-KL PPO，F8 只写成理论小节。

---

## 5. 第 1 天结果（2026-10-30，5090d）

**环境**：Python 3.11、torch 2.8.0+cu128（CUDA 12.8，能识别 RTX 5090 D 的 sm_120）、mani_skill 3.0.1、sapien 3.0.3。

注意：SAPIEN 第一次启用 GPU 物理引擎时，要从 GitHub 下载 PhysX 的 GPU 库。5090 直连下载会失败，要走本机代理 `127.0.0.1:17890`。这个库现在已经放在 `~/.sapien/physx/105.1-physx-5.3.1.patch0/`。

**单任务吞吐**：在循环里带一个 256 宽的小策略网络，不含训练开销。

| 任务 | 并行环境数 | 环境步/秒 | 显存 |
|---|---|---|---|
| PushCube | 1024 | 75,543 | 2.25 GiB |
| PushCube | 2048 | 133,947 | 2.44 GiB |
| PushCube | 4096 | 219,226 | 2.76 GiB |
| PickCube | 2048 | 126,799 | 2.44 GiB |
| PegInsertionSide | 1024 | 25,317 | 2.32 GiB |

**状态恢复**：存状态、恢复、再用同一串随机动作重放 30 步，1024 个环境。
- PushCube：98.3% 的环境方块位置差小于 1 mm，平均差 0.44 mm；
- PickCube：98.1% 小于 1 mm，平均差 0.29 mm。
- 少数环境在随机乱动时会发散得很厉害。随机动作几乎不会成功，所以「结局能否复现」还没测到，要等有了熟练策略再测。

**统一观测的多任务环境**：9 个任务映射到同一套 78 维槽位：
- 机器人本体 25 维；
- 被搬物体 11 维、工具 12 维、目标物体 11 维，都包含位姿、半尺寸、是否被抓住、是否存在；
- 目标位置 4 维；
- 相对向量 6 维；
- 任务独热编码 9 维。

在一个进程里同时开 9 个 GPU 仿真，每个任务 256 个环境，随机动作下约 1.15 万步/秒，显存 11.1 GiB。这远低于单任务的速度，因为 9 个仿真依次各跑一小批，调度开销占了大头。训练源策略前，先试「每个任务 1024 个环境、不含 PegInsertionSide」的配置；如果还是慢，就改成每个任务一个进程。

---

## 6. 文件

| 文件 | 作用 |
|---|---|
| `scripts/setup_5090.sh` | 5090 上的环境：Python 3.11、torch 2.8.0（PyPI 版本即 CUDA 12.8）、mani_skill |
| `scripts/smoke_maniskill.py` | 吞吐和显存的冒烟测试 |
| `scripts/restore_check.py` | 状态恢复保真度测试：存状态、恢复、重放同一串动作，比较结局 |
| `scripts/inspect_obs.py` | 列出各任务的观测字段 |
| `wrwl/envs/unified.py` | 统一观测的适配层，以及多任务向量环境 |
| `scripts/test_unified.py` | 多任务环境的冒烟测试：形状、有限值、槽位掩码、吞吐 |
