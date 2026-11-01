# 运行记录

这里是各台机器上每次运行留下的小文件，拷贝时间是 2026-11-01 18:50–19:03。

- 目录结构是 `results/<机器>/<代码目录>/{runs,logs,smoke}/…`，和机器上的路径一一对应。
  - 机器：haha-a、haha-c、haha-d 是 RTX 4090，5090d 是 RTX 5090 D。
  - 代码目录：各目录对应的代码版本见 [../deploy/README.md](../deploy/README.md)。
- 只拷了文本：每次运行的 `args.json`（完整参数和 `code_md5`）、`eval.csv`（每 102,400 步一次评估）、`train.csv`、训练日志 `logs_<run>.txt`、保持率评估的 `ret/*.json`，以及队列日志。模型权重没有拷。
- 结论和判定规则写在 [../docs/conflict-reuse-2026-10-31.md](../docs/conflict-reuse-2026-10-31.md)：每一节都注明了读的是哪些运行、按哪条事先登记的规则判定。
- 拷贝时还在跑的：5090d 上的机制实验（`5090d/zhr_pve3/runs/why`）；haha-c 上 FN-PPO 的放置复现刚开始。之后的结果会在后续提交里补上。
