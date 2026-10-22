# Day 7｜BPR-MF（LearnBPR SGD + Adam 对照）记录模板

## 1. 原理与实验边界

1. 为什么 **BPR-Opt** 不规定优化器，而原论文的 **LearnBPR** 使用带放回采样与 SGD？
2. 区分三元组均匀抽样与「每个正交互采一次负样本」；写出用户 u 的抽样权重：
3. 为什么 `batch_size=1` 比 `batch_size=512` 更接近原论文的逐三元组 SGD？
4. 为什么 SGD 与 Adam 不能直接套用相同的最佳学习率？
5. 为什么验证/测试数据不能用来构造训练正例集合？
6. 为什么训练 Loss 降低不等于 Recall@K 一定上升？

## 2. 实验约定

- 数据版本：MovieLens-100K / Synthetic
- 正交互定义：所有观察到的评分为正交互（Day 7 教学约定）
- 种子：20261020
- 数据划分：train / validation / test = 近似 80% / 10% / 10%，按用户划分
- 模型：BPR-MF、无偏置、Embedding 初始化 N(0,0.1²)
- 优化目标：mean(softplus(score_neg - score_pos)) + λ · mean(Batch Embedding 三向量平方和)
- 采样：均匀抽取合法三元组 D_S、带放回
- 说明：Mini-batch=512 为工程变体，Mini-batch=1 为逐三元组更新

## 3. SGD 主实验

- Learning Rate：
- Embedding Dim：
- Batch Size：
- Steps per Epoch：
- Epochs：
- 初始权重 SHA256：
- 第 1 轮三元组抽样 SHA256：
- Train / Val / Test 数量：
- 首轮 BPR Loss：
- 最后 BPR Loss：
- Pairwise Accuracy（描述候选集及抽样方案）：

## 4. Adam 对照实验

- Learning Rate（须单独选取）：
- 初始权重 SHA256：
- 第 1 轮抽样 SHA256：
- 与 SGD 两个 Hash 是否一致：
- Adam 首轮/最后 BPR Loss：
- 其它超参数是否一致：
- 仅凭训练损失能否断言推荐质量优劣：

## 5. 文件与公平性审计

- `config.json`、`history.jsonl`、`checkpoint.pt`：
- `train_pairs.tsv`、`valid_pairs.tsv`、`test_pairs.tsv`：
- 三份集合是否两两不重叠：
- 负采样是否只使用 train：
- 随机采样存在重复三元组的检查：
- 是否存在任何可采负例为零的用户；如何处理：

## 6. Day 7 自动验收

- 不提供 MovieLens 的 A–E： /80
- 提供官方 MovieLens 后 A–F： /100
- 最难排查的 bug：
- Day 8 的评估准备：
