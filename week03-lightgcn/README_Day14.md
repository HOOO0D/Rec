# Day 14｜LightGCN 训练与评价练习包

此包按照 `RoadMap(2).md` 的 Day 14 编写，承接 Day 7、Day 8、Day 11、Day 13，保持每日 4–5 小时、任务—范围—验收的教学风格。

## 文件结构

| 文件 | 用途 | 是否需要修改 |
|---|---|---|
| `lightgcn_train.py` | 负采样、BPR Loss、单轮训练、Full Ranking 的中文 TODO 模板 | **是**：独立实现 A–D |
| `Test_Day14.py` | 自动验收脚本（A–F，100 分） | 否 |
| `run_day14.py` | 真实 MovieLens 训练与验证的完整执行器（不包含 TODO 答案） | 否 |
| `Day14_notes.md` | 公式、理论题、协议审计和实验记录 | **是**：填写 |
| `README_Day14.md` | 本文件 | 否 |

**注意：** 为保证独立练习，本包不提供已完成的 `lightgcn_train.py` 参考实现。请复用你自己在 Day 13 完成的 `lightgcn_model.py`，不要用本包覆盖 Day 13 的原文件。

## 依赖和推荐路径

使用此前的 `torch-learning` 环境，依赖 PyTorch 即可；不需要安装 DGL、PyTorch Geometric 或 RecBole。

将本包所有文件复制到：

```text
E:\AI\rec-learning\week03-lightgcn\
├── lightgcn_model.py             # 你在 Day 13 已完成
├── lightgcn_train.py             # 本日完成 TODO
├── Test_Day14.py
├── run_day14.py
├── Day14_notes.md
└── experiments\
    ├── day10_ml100k\
    ├── day11_ml100k\
    │   ├── normalized_adjacency.pt
    │   └── normalization_summary.json
    └── day14_ml100k\
```

Day 7 冻结数据目录需包含：

```text
config.json
train_pairs.tsv
valid_pairs.tsv
test_pairs.tsv
```

**不要重新生成数据划分**。训练图和训练交互应来源于同一个冻结 Day 7 实验。

## 测试 A–E：80 分

在 PowerShell 中运行：

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week03-lightgcn
python .\Test_Day14.py
```

此时 F 项会标记为 SKIP，因此满分为 80/100。

## 测试 F：完整 100 分

`day7_sgd_ml100k` 是示例目录名，请按你真实的 Day 7 保存位置修改：

```powershell
python .\Test_Day14.py `
    --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k `
    --graph-dir .\experiments\day11_ml100k
```

F 项会检查图源、训练集合和边权，然后运行 **1 轮** LightGCN 训练，输出真实验证 Recall@10、Hit@10、NDCG@10。

## 真实实验：先 Smoke，再正式训练

```powershell
# 一轮连通性测试
python .\run_day14.py `
    --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k `
    --graph-dir .\experiments\day11_ml100k `
    --output-dir .\experiments\day14_smoke `
    --epochs 1 --batch-size 2048 --device cpu

# 正式教学基线（RTX 4060 可用时使用 CUDA）
python .\run_day14.py `
    --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k `
    --graph-dir .\experiments\day11_ml100k `
    --output-dir .\experiments\day14_ml100k `
    --embedding-dim 16 --n-layers 2 `
    --batch-size 1024 --epochs 10 `
    --learning-rate 0.001 --reg-lambda 0.0001 `
    --topk 10 --seed 20261031 --device cuda
```

如果当前环境的 `torch.cuda.is_available()` 为 `False`，把 `--device cuda` 改为 `--device cpu`。这份示例超参数仅用于教学，并非 LightGCN 的最优设置。

## 最终输出

```text
experiments/day14_ml100k/
├── experiment_config.json       # 运行配置、协议、图源 SHA256
├── epoch_metrics.csv            # 每轮训练 Loss 与验证 Top-K 指标
├── best_valid_model.pt          # 由验证 NDCG@10 选择的最佳模型
└── run_summary.json             # 最佳验证结果和训练摘要
```

Day 14 **不会输出最终测试集推荐指标**；测试集只用于数据划分冲突审计。Day 15 做模型比较和最终测试时再使用。

## 常见问题

- `NotImplementedError`：你还没有完成 `lightgcn_train.py` 对应 TODO，或 Day 13 的模型尚未实现。
- `train_pairs_sha256 不匹配`：Day 7 的训练交互与 Day 11 图来自不同数据划分，不能混用。
- `图中出现非训练边`：图构造可能发生数据泄漏，需要回溯 Day 10 的图源。
- `CUDA 不可用`：选择 `--device cpu`，或检查原来的 Conda 环境是否安装 CUDA 版 PyTorch。
- `训练 Loss 降低，但验证 NDCG 不升`：先检查评估规则、负采样、正则化、模型保存和随机种子；不要直接调测试集。

## 参考

- LightGCN 论文 Section 3.3：<https://hexiangnan.github.io/papers/sigir20-LightGCN.pdf>
- PyTorch Softplus：<https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html>
- PyTorch sparse.mm：<https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html>
