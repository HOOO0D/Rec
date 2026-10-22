# ML-100K 优化器和训练目标对比

独立脚本 `plot_ml100k_comparison.py` 复用现有模型、采样器、数据划分和 BPR 训练函数，不改动这些文件。

## 运行

在 PowerShell 中：

```powershell
cd E:\AI\rec-learning\week02-bpr
$py = "E:\AI\envs\torch-learning\python.exe"
& $py -B plot_ml100k_comparison.py --epochs 30
```

自动查找脚本目录下的 `ml-100k/u.data` 或 `data/ml-100k/u.data`。也可以显式传入：

```powershell
& $py -B plot_ml100k_comparison.py --data-path .\ml-100k\u.data --epochs 50 --output-dir .\runs\ml100k_comparison_50
```

输出目录必须尚不存在，避免覆盖已有实验。重复运行时请指定一个新的 `--output-dir`。

依赖：PyTorch、NumPy、Matplotlib。当前 torch-learning 环境均已安装。默认 CPU，单线程；可用 `--threads` 设置线程数。`--device cuda` 要求支持 CUDA 的环境。

## 四组方法

| 方法 | 目标 | 默认学习率 |
|---|---|---:|
| BPR-SGD | 成对 logistic 损失 + 批内嵌入 L2 | 0.20 |
| BPR-Adam | 成对 logistic 损失 + 批内嵌入 L2 | 0.01 |
| MF-SGD | 已观察评分 MSE + 全表 L2 | 0.05 |
| MF-Adam | 已观察评分 MSE + 全表 L2 | 0.01 |

四组使用同一 80%/10%/10% 用户内划分、维度、批次大小和初始嵌入。两组 BPR 使用相同逐轮采样序列并检查 SHA256。两组 MF 使用相同逐轮乱序。BPR 每轮默认抽取训练交互数量的三元组；MF 每轮遍历一次训练评分，均分为批次更新。

MF 复用 `model.py` 中的 `MatrixFactorization` 和 `mf_loss`；为了匹配批次规模，脚本提供小批次训练循环，原有 `train_mf` 的全批量训练函数保持原样。初始权重统一为 BPR 的 N(0,0.1²)，替代 MF 默认初始化，以控制初始化差异。

默认参数是可复现的教学基线，未经过最优学习率搜索。尤其小批次平均损失下，SGD 的更新可能较慢。可以仅根据验证结果修改下列参数，并使用新输出目录重跑：

```text
--bpr-sgd-lr  --bpr-adam-lr  --mf-sgd-lr  --mf-adam-lr
--bpr-l2      --mf-reg       --batch-size --embedding-dim
--epochs      --steps-per-epoch
```

当自定义 `steps-per-epoch` 时，只改变 BPR 的采样量，MF 仍遍历全部训练评分；相同 epoch 不代表相同运行时间。

## 曲线和产物

- `validation_metrics.png`、`.svg`：AUC、Recall@K、NDCG@K、Precision@K、HitRate@K，以及仅对 MF 有意义的评分 RMSE。
- `training_loss.png`、`.svg`：每个 epoch 的训练损失，实线包括正则项，虚线只含数据损失。BPR 和 MF 的目标尺度不同，因此分别画图。
- `metrics.csv`：四组方法的逐轮验证指标和损失。epoch 0 是初始化评估，没有训练损失。
- `config.json`：参数、数据与源码摘要、评估约定、初始权重摘要。
- 三份 `*_pairs.tsv`：实际训练、验证和测试划分。
- 每组目录的 `history.jsonl`：训练与验证逐轮日志，BPR 包含采样摘要。
- 每组目录的 `final.pt`：最终模型、优化器状态和配置。
- 每组目录的 `best.pt`：按验证 AUC 选择的最佳模型，不包含优化器状态，用于评估而非原样恢复训练。
- `summary.json`、`report.md`：所选模型在测试集上的比较表。

## 指标解释

每轮只画验证曲线；按验证 AUC 最大的 epoch 选择模型，同分取最早的训练 epoch，之后只评估一次测试集。

排序相关性沿用 Day 7：所有已观察评分，包括低评分，都视为正交互。训练仅使用训练划分。验证和测试的候选集合分别为“当前留出集正例 + 该用户在整个数据集中从未观察的所有物品”，不抽样负例。其他已观察交互被排除，因此验证与测试不会互相作为负例；全数据中的已观察集合只用于构造评估候选，不进入训练采样器。未观察不等于实际不喜欢，这是隐式数据评估的代理标签。

指标先在每位用户上计算，再做宏平均：

- AUC：随机选一个留出正例和一个候选负例，正例得分更高的概率；同分计 0.5，精确计算全部正负比较。
- Recall@K：前 K 个推荐命中的正例数 / 当前用户全部留出正例数。
- Precision@K：前 K 个推荐命中的正例数 / K；候选不足 K 时分母使用实际数量。
- NDCG@K：对命中位置作对数折扣，再除以该用户理想排序的折扣得分，使用二元相关性。
- HitRate@K：前 K 个推荐是否至少命中一个正例。
- RMSE、MAE：MF 对留出集 1～5 分评分的预测误差，不截断预测。BPR 输出是排序分数，不比较其评分误差。

Top-K 的相同分数按物品 ID 升序打破平局。初始化的同分参考曲线可能因该规则有微小差别。单个种子、固定参数的曲线不能说明某个优化器普遍优于另一个。

定义参考：[ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics/roc_auc_score.html)、[NDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics/ndcg_score.html)。
