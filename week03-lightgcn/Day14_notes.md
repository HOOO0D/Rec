# Day 14｜LightGCN 完整训练与 Full Ranking 验证

> 本文件是当天需要填写的学习与实验记录，建议按题目独立作答。
>
> 基于 RoadMap(2).md Day 14，仍沿用 Day 7 冻结数据、Day 11 训练图和 Day 13 LightGCN 模型。

## 1. 今日目标与边界

- 目标：连接 BPR Loss、负采样、Adam 和 Full Ranking 评价，完成简化版 LightGCN。
- 今天：小图能够训练、梯度与参数有限、输出验证 Recall@10 / Hit@10 / NDCG@10、保存配置与 Checkpoint。
- 暂不做：Day 15 的 BPR-MF vs LightGCN 公平对比、`n_layers=0…4` 消融和根据测试集指标调参。

## 2. 公式复习

初始用户/物品可训练参数：

\[
\Theta=\{P^{(0)},Q^{(0)}\}
\]

归一化图传播：

\[
E^{(k+1)}=\widetilde A E^{(k)}, \quad \widetilde A=D^{-1/2}AD^{-1/2}
\]

层融合：

\[
E_{final}=\frac1{K+1}\sum_{k=0}^K E^{(k)}
\]

正负评分：

\[
\hat y_{ui}=e_u^T e_i,\qquad \hat y_{uj}=e_u^T e_j
\]

训练目标：

\[
L_{rank}=\frac1B\sum_{(u,i,j)\in\mathcal B}\operatorname{softplus}(\hat y_{uj}-\hat y_{ui})
\]

本教学实现使用 mini-batch 第 0 层三组 Embedding 正则化：

\[
L_{reg}=\frac1{2B}\sum_{(u,i,j)\in\mathcal B}
(\|p_u^{(0)}\|^2+\|q_i^{(0)}\|^2+\|q_j^{(0)}\|^2)
\]

\[
L_{total}=L_{rank}+\lambda L_{reg}
\]

说明：论文 Section 3.3 将正则化表述为 \(\lambda\|E^{(0)}\|^2\)；本次采用显式定义的 Batch 估计形式。不同的 L2 归一化定义与训练采样分布会影响实验，不能在比较时忽略。

## 3. 理论题（完成后检查）

1. 为什么 BPR 正负预测必须采用传播融合后的 Embedding，而不是直接拿初始 Embedding 评分？
   - 我的回答：
2. 为什么正则项只作用于第 0 层 Embedding？
   - 我的回答：
3. 如果一个用户所有物品都已在训练集中出现，负采样要怎么处理？
   - 我的回答：
4. 为什么用户的验证正例可能在训练负采样时被选中？这一定是“泄漏”吗？
   - 我的回答：
5. 为什么每个训练 Batch 都要重新调用图传播？能否保存一次最终 Embedding 然后所有 Batch 共享？
   - 我的回答：
6. 为什么运行 `evaluate_full_ranking()` 时要 `eval()` 和 `torch.no_grad()`？
   - 我的回答：
7. Recall@K、Hit@K、NDCG@K 的单用户定义和宏平均是什么？
   - 我的回答：
8. 验证集、测试集分别允许在本日哪些步骤被读取？
   - 我的回答：
9. 为什么 `weight_decay=0.0` 时，仍可能出现模型正则化效果？
   - 我的回答：
10. 模型训练 Loss 降低但验证 NDCG 下降，可能是什么原因？
    - 我的回答：

## 4. Day 7 / Day 11 / Day 13 兼容性审计

- [ ] Day 7 `config.json` 记录了 `n_users/n_items/n_train/n_valid/n_test`。
- [ ] Day 7 的 `train_pairs.tsv/valid_pairs.tsv/test_pairs.tsv` 互不重叠、均为内部零基 ID。
- [ ] Day 11 的 `normalization_summary.json` 中 `graph_source=train_only`。
- [ ] `train_pairs_sha256` 与 Day 7 的训练集精确一致。
- [ ] Day 11 图的每一条无向边都对应训练正交互。
- [ ] Day 13 LightGCN 的归一化邻接矩阵是 buffer，不是可训练权重。
- [ ] 全物品排序时屏蔽了训练已见物品，而没有屏蔽当前验证集正例。

## 5. 实验记录（实际运行后填写）

- 实验日期：
- Day 7 冻结数据目录：
- Day 11 归一化图目录：
- 训练图 SHA256：
- PyTorch 版本：
- CUDA 版本：
- Device：
- Embedding 维度 / LightGCN 层数：
- Optimizer / LR / Batch Size / Epochs：
- 负采样分布：
- L2 正则化定义 / 系数：
- 训练 Loss 首轮 / 末轮：
- 验证 Recall@10：
- 验证 Hit@10：
- 验证 NDCG@10：
- 最佳验证 Epoch：
- Checkpoint 文件：
- 是否使用测试指标选模型：**否**
- 已知协议与 Day 7 手写 BPR 的差异：
- 下一步（Day 15）应控制的变量：

## 6. Day 14 必须完成的交付物

- `lightgcn_train.py` 的全部 TODO。
- `Test_Day14.py` 100/100（其中 F 必须使用真实 MovieLens 冻结数据）。
- `experiments/day14_ml100k/experiment_config.json`
- `experiments/day14_ml100k/epoch_metrics.csv`
- `experiments/day14_ml100k/best_valid_model.pt`
- `experiments/day14_ml100k/run_summary.json`

## 7. Day 15 预告

只有先验证 BPR-MF 与 LightGCN 使用**相同的数据划分、候选集合、已见物品过滤、负采样、指标实现与必要的超参数口径**，才做模型对比。再固定实验条件，比较 LightGCN 的 `n_layers=0/1/2/3/4`。测试指标在最终模型选择之后报告。
