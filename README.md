Where to Reuse, Where to Learn

往同一个机器人策略网络里持续加新技能，同时不忘旧技能。环境是 ManiSkill3，状态输入，用 PPO 训练。主方法是零空间适配器（null-space adapter）：旧技能的输出层读不到适配器写进去的东西，所以旧技能按构造精确保持不变；新技能仍能复用旧网络学到的表示。

- `wrwl/`：训练代码（`ppo.py` 是训练入口，`nsa.py` 是零空间适配器，`envs/` 是新任务的环境）
- `scripts/`：排队、评估、分析和判定脚本
- `docs/`：实验结论和判定规则。主文件是 [docs/conflict-reuse-2026-10-31.md](docs/conflict-reuse-2026-10-31.md)，按时间倒序记录，每条结论都先登记判定规则、再看结果。`docs/figures/` 是论文主图。
- `deploy/`：各台机器上实际跑过的代码快照，以及每版代码的 md5
- `results/`：每次运行的参数、评估曲线和日志（不含模型权重）
- `third_party/ms_ppo/`：ManiSkill 官方 PPO 基线的拷贝（用来读取专家模型）
