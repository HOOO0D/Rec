# Day 7 验收（修订版）：LearnBPR 的 SGD 与 Adam 对照

**本次已修正原 Day 7：** 旧版默认 Adam、每条正交互配一条负例的采样方式，不应被称为论文原始 LearnBPR。新版要求以 `SGD + 均匀有放回 D_S 抽样` 为主实验；`Adam` 在相同初始化、抽样与划分下作为对照。

原论文：Rendle et al. (2009), *BPR: Bayesian Personalized Ranking from Implicit Feedback*, Sec. 4.2–4.3.1. https://proceedings.mlr.press/r7/rendle09a.html

## 文件 / 如何覆盖旧版本

将本练习包中的 `bpr_model.py`、`negative_sampling.py`、`train.py`、`Test_Day7.py`、`compare_runs.py`、`Day7_notes.md` 解压到 `E:\AI\rec-learning\week02-bpr\`，**同名文件应先备份，再覆盖**。不要覆盖 Day 5 的 `model.py` 或 Day 6 的 `bpr_practice.py`。新版测试接口与旧版不兼容，请不要混用。

源码保留 TODO，未包含完整模型、采样器、训练器实现。运行测试之前必须自己完成 TODO。

## LearnBPR 抽样的核心约束

对于用户 u，已有训练正交互集合 `Iu`，物品总数 I：

`D_S = {(u,i,j) | i in Iu, j not in Iu}`。

第 u 个用户的合法三元组数为 `w_u = |Iu| * (I - |Iu|)`，因此**均匀抽整个 D_S** 应按 `w_u` 加权选择用户，再对其正负物品均匀采样。用户全正或全无观察时，权重为 0。

```text
训练数据 ──> 正交互集合 ──> 均匀 BootstrapSampler ──> (u,i,j) ──> BPRMF
                                                             |
                                                             v
                                             softplus(s_neg - s_pos) + L2
                                                             |
                                                             v
                                                 torch.optim.SGD.step()
```

抽取三元组 **有放回**，因此 `(u,i,j)` 可以重复。若每条训练正交互都恰好采一次负物品，即使负物品正确，也**不是均匀抽取 D_S**（不同用户未观察候选数量可能不同）。

原论文逐三元组 SGD 对应 `batch_size=1`。为了让 ML-100K 的 CPU 训练可接受，可用 `batch_size=512`：这是保留相同采样分布的 **Mini-batch SGD 工程变体**，不是逐步完全一致的论文算法。

## 验收评分

| 测试 | 内容 | 分值 |
|---|---|---:|
| A | 模型打分、BPR Loss、数值与梯度 | 15 |
| B | Bootstrap 三元组合法性、均匀性、可复现及满交互边界 | 20 |
| C | 用单一三元组核对 SGD 更新的具体数值 | 20 |
| D | SGD + Bootstrap 在合成数据上有效收敛（固定候选） | 15 |
| E | Adam 对照：相同初始参数、相同抽样顺序，日志可追溯 | 10 |
| F | 使用真实 MovieLens-100K 进行至少 1 Epoch 的 SGD 训练，留出数据并落盘 | 20 |
| **总计** | **6/6 全部通过** | **100** |

没有真实 MovieLens-100K `u.data` 时只能做 A–E，最多 80/100；不应把结构正确的模拟数据冒充官方数据。

## 环境与运行方法

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week02-bpr
python .\Test_Day7.py
```

先完成模板所有 TODO。然后下载官方 MovieLens-100K https://grouplens.org/datasets/movielens/100k/ ，解压 `ml-100k/u.data` 至 `week02-bpr/data/ml-100k/u.data`。

```powershell
# 一次完整验收（含真实 MovieLens-100K）
python .\Test_Day7.py --ml100k .\data\ml-100k\u.data

# SGD 主实验：相对接近原论文的抽样和单样本更新
python .\train.py --dataset synthetic --optimizer sgd --lr 0.20 --epochs 50 --steps-per-epoch 216 --batch-size 1 --output-dir .\runs\day7_sgd_single

# 为更快的 ML100K 训练，使用 mini-batch SGD
python .\train.py --dataset ml100k --data-path .\data\ml-100k\u.data --optimizer sgd --lr 0.05 --epochs 1 --batch-size 512 --output-dir .\runs\day7_sgd_ml100k

# 公平对照：相同 seed、steps-per-epoch、batch-size、维度，优化器学习率单独选取
python .\train.py --dataset synthetic --optimizer sgd --lr 0.20 --epochs 50 --steps-per-epoch 216 --batch-size 8 --seed 20261020 --output-dir .\runs\day7_sgd_compare
python .\train.py --dataset synthetic --optimizer adam --lr 0.02 --epochs 50 --steps-per-epoch 216 --batch-size 8 --seed 20261020 --output-dir .\runs\day7_adam_compare
python .\compare_runs.py .\runs\day7_sgd_compare .\runs\day7_adam_compare
```

自动测试的合成 SGD 超参数是该测试专用的固定配置，不代表真实数据上的最佳配置。**不要为了指标反复使用测试集选超参数**。正式比较优化器的泛化性能，须等 Day 8 实现同一候选集下的 Recall@K / NDCG@K 后，在验证集上完成选择。

## 实验产物

每一组 `run_experiment` 都应保存：

- `config.json`：包含模型维度、种子、优化器、学习率、采样方式、Epoch / Steps / Batch、正则化约定、数据量、初始参数 SHA256。
- `train_pairs.tsv`、`valid_pairs.tsv`、`test_pairs.tsv`：零基 ID、互不交叉，供 Day 8 使用；不要用验证/测试交互构造训练正样本。
- `history.jsonl`：每轮 BPR Loss、总 Loss、三元组数量、固定种子采样 SHA256，以及当前样本上的成对正确率。
- `checkpoint.pt`：模型、优化器 state_dict 及配置。

训练阶段的 `pairwise_acc` 随动态采样变化，不是严格可比的验证指标。合成数据验收 D 另外计算**固定全部合法三元组**的成对正确率，避免将随机采样波动当作收敛证据。

## 理论验收（手工检查）

`Day7_notes.md` 中的论文原理、采样分布、单步 SGD 更新、实验公平性和数据泄漏问题必须独立回答。100 分代码通过不意味着全部理论验收通过。完整复现论文全部实验结果不属于本日目标。
