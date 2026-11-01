# 实际跑过的代码版本

实验在实验室的几台机器上跑，每台机器用一个独立的代码目录。这里存的是这些目录在 2026-11-01 19:00 的原样快照：`wrwl/` 是训练代码，`scripts/` 是排队、评估和分析脚本，顶层的 `drv_*.sh` 是当天用的队列脚本。模型权重和数据文件（`*.pt`）没有放进仓库。

每次运行都会把 `wrwl/` 下全部 `.py` 文件（按路径排序后拼接）的 md5 写进自己的 `args.json`，字段名是 `code_md5`。用这个值就能查到某次运行用的是下面哪一版代码。

| 目录 | 机器上的位置 | 运行记录的 code_md5 | 用途 |
|---|---|---|---|
| `haha-a_zhr_pve2/` | haha-a `~/zhr_pve2` | `7dcfdd0b…` | 主代码：投影消融（`runs/protect_np`）、技能链、从零训练的队列（`drv_a6`–`drv_a8`） |
| `haha-c_zhr_pve2/` | haha-c `~/zhr_pve2` | `7dcfdd0b…` | 同一份主代码：FN-PPO、整网微调和只训输出层的放置对照（`drv_c5`–`drv_c8`） |
| `zhr_pve3/` | haha-c、haha-d、5090d 上的 `~/zhr_pve3` | `90142222…` | 机制实验（`scripts/run_why.sh`，判定规则在 `scripts/why_test.py`）；5090d 的环境搭建和队列脚本也在这里 |
| `zhr_pve4/` | haha-c、haha-d 上的 `~/zhr_pve4` | `cc8d7f41…` | 任务组：主代码加上新的环境，以及 `scripts/suite_*` |

- 仓库根目录的 `wrwl/` 与 `haha-a_zhr_pve2/wrwl/` 完全相同（code_md5 `7dcfdd0b…`）。
- 更早的运行（10-30、10-31）记录的是别的 code_md5，那些版本的代码没有单独保存。
- `zhr_pve3/with_gpu_mem_5090d.sh` 是 5090d 从 11-01 18:58 起用的版本：启动锁从 90 秒缩到 30 秒，其余不变。它只影响任务什么时候开始，不影响训练本身，也不进入 code_md5。
