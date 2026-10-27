# Day 8｜BPR-MF Top-K 评价练习包

## 本日定位

这是 RoadMap 第 2 周 Day 8 的独立实现练习，承接 Day 7 的 `SGD + Bootstrap` BPR-MF 模型与冻结的 MovieLens-100K Train/Validation/Test 划分。

**不重新训练模型，不使用 RecBole 内置评价器，不根据测试集调参。**

课程依据：`RoadMap.md` 中的「Day 8｜评价指标」及其验收标准。补充参考：
- https://recbole.io/docs/recbole/recbole.evaluator.metrics.html
- https://www.recbole.io/docs/user_guide/config/evaluation_settings.html
- https://www.d2l.ai/chapter_recommender-systems/ranking.html

## 文件清单

| 文件 | 内容 |
|---|---|
| `evaluate.py` | 本日 7 个独立实现函数，保留中文 TODO 和中文文档注释 |
| `Test_Day8.py` | 6 组自动测试，A–E 共 80 分，F 为真实实验集成 20 分 |
| `run_day8.py` | 读取 Day 7 实验目录、恢复 BPR-MF、记录验证集指标；无需补 TODO |
| `Day8_notes.md` | 手算题、指标公式、理论验收与实验记录 |
| `README_Day8.md` | 本说明 |

## 放置目录

```text
E:\AI\rec-learning\week02-bpr\
├── bpr_model.py               # 你在 Day 7 已实现的模型
├── negative_sampling.py       # Day 7 已完成
├── train.py                   # Day 7 已完成
├── evaluate.py                # Day 8，新建
├── run_day8.py                # Day 8，新建
├── Test_Day8.py               # Day 8，新建
├── Day8_notes.md              # Day 8，新建
└── runs\
    └── day7_sgd_ml100k\
        ├── config.json
        ├── train_pairs.tsv
        ├── valid_pairs.tsv
        ├── test_pairs.tsv
        ├── history.jsonl
        └── checkpoint.pt
```

不要覆盖 Day 7 的 `bpr_model.py` 或 `train.py`。

## 运行指令（PowerShell）

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week02-bpr

# 先完成 evaluate.py 所有 TODO，再验证 A–E：
python .\Test_Day8.py

# 已有真正的 Day 7 MovieLens-100K 产物后，执行完整验收：
python .\Test_Day8.py --run-dir .\runs\day7_sgd_ml100k

# 正式保存验证集结果：
python .\run_day8.py --run-dir .\runs\day7_sgd_ml100k --output-dir .\runs\day8_sgd_valid --k 10
```

也支持 `ml100k_sgd_lr5` 这类多模型对比实验目录。配置和三份划分仍从根目录读取；默认加载 `BPR-SGD/best.pt`：

```powershell
python .\Test_Day8.py --run-dir .\runs\ml100k_sgd_lr5
python .\Test_Day8.py --run-dir .\runs\ml100k_sgd_lr5 --method BPR-Adam
python .\Test_Day8.py --run-dir .\runs\ml100k_sgd_lr5 --checkpoint BPR-SGD/final.pt
python .\run_day8.py --run-dir .\runs\ml100k_sgd_lr5 --output-dir .\runs\day8_lr5_valid
```

原 Day 7 目录默认加载根目录的 `checkpoint.pt`。`--checkpoint` 可以指定权重文件，其相对路径以 `--run-dir` 为基准，并优先于默认选择。输出指标和评价配置会记录实际加载的权重路径、训练轮次及优化器来源。

对比实验中的 `best.pt` 保存验证集 AUC 最高轮次的模型权重与轮次；`final.pt` 保存最后一轮的模型权重、优化器状态和配置。最高 AUC 不保证最高 Recall/NDCG；训练可能在后续轮次退化，因此 best 与 final 可能不同。两者都能用于本日评价。

## 评价协议的固定约定

- Full Ranking：对每位有验证正例的用户，对物品全集 `[0,n_items)` 评分，而不是随机抽 100 个负物品。
- 验证阶段屏蔽该用户**训练已交互物品**，验证正例不能屏蔽；测试集不参与评价与挑选超参数。
- 物品评分由高到低；同分时物品 ID 升序，使结果稳定可复现。
- K 为正整数；有效候选不足 K 时只返回可推荐候选，不用已交互物品填充。
- 每位有验证正例的用户计算 Recall@K、Hit@K、NDCG@K，再做**用户宏平均**；没有验证正例的用户不计入分母。
- 二值相关性；NDCG 的 IDCG = `sum(1/log2(rank+1), rank=1..min(K,正例数))`。
- `evaluate.py` 的单用户指标函数遇到空正例集合约定返回 0.0；完整评价器没有任何验证正例则应抛出 `ValueError`。
- 训练已交互屏蔽由**评分克隆**实现，不能原地改动模型输出；被屏蔽物品的分数设为负无穷并从 Top-K 排除。

## 验收打分

| 分组 | 内容 | 分数 |
|---|---|---:|
| A | 手算指标、空正例和 K 边界 | 20 |
| B | 训练已交互屏蔽及不修改原评分 | 15 |
| C | Top-K 稳定排序、屏蔽和候选不足 | 15 |
| D | Train/Valid/Test 互不重叠及去重 | 15 |
| E | 玩具模型全排序、用户宏平均、无梯度评价 | 15 |
| F | 真实 MovieLens-100K Checkpoint 的验证集评价和结果落盘 | 20 |
| **合计** | **6/6 PASS** | **100** |

没有 `--run-dir` 时仅执行 A–E，最多 80/100，会提示未完成集成验收。F 项会核对实验目录的用户/物品数量与 100000 条数据；无法单凭元数据证明文件确为官方数据，必须由你确认数据来源。

## Day 8 输出与后续使用

正式运行后保存到 `runs/day8_sgd_valid/`：
- `validation_metrics.json`：Recall@K、Hit@K、NDCG@K、评价用户数及来源优化器。
- `evaluation_config.json`：全排序、过滤规则、K、种子和运行目录等。
- `validation_recommendations.tsv`：前 10 位验证用户的 Top-K 推荐列表，供人工检查训练过滤。

对于 Adam 对照实验，用相同参数执行一遍并指定另一个输出目录。**没有固定同一数据切分、候选集和过滤规则时，不能直接比较指标。**

## Day 9 预备

Day 9 使用 RecBole 运行 BPR 对照。首先核对数据划分、候选模式（Full Ranking 或采样）、已交互过滤、指标定义和正例转换，然后再解释指标是否存在差异。
