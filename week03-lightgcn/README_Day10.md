# Day 10｜图论、二部图与邻接矩阵构造

本练习包按照 `RoadMap.md` 的 Week 3 / Day 10 编写，承接 Day 7 的 BPR-MF 实验数据；代码模板和测试脚本使用中文注释。

**目标**：正确构造用户—物品二部图及邻接矩阵，为 Day 11 的邻接矩阵归一化与 Day 13 的 LightGCN 图传播做准备。

## 一、下载与放置

将练习包中的四个文件放到：

```text
E:\AI\rec-learning\week03-lightgcn\
├── graph_basics.py         # 六个 TODO：请独立实现
├── Test_Day10.py           # 自动验收，A–F 共 100 分
├── run_day10.py            # Day 7 实验数据集成的可执行脚手架
├── Day10_notes.md          # 手算、阅读范围、理论与实验记录
└── README_Day10.md         # 说明文件
```

不要覆盖你原先 `week02-bpr` 中已经完成的 `train.py`、`evaluate.py` 或数据划分。

## 二、环境

使用之前能运行 PyTorch 的 Conda 环境即可；无须安装 NetworkX、PyG、DGL 或 RecBole：

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week03-lightgcn
python -c "import torch; print(torch.__version__)"
```

可在 GPU 环境使用本练习，不过本日图构造和验收默认在 CPU 上运行；MovieLens-100K 规模足以在普通 PC 上完成。

## 三、执行顺序

先手算，再填写 `graph_basics.py` 的 TODO：

```powershell
# 先测 A–E：正确后只能得到 80 分
python .\Test_Day10.py

# 使用 Day 7 冻结的真实 MovieLens-100K 数据完成 100 分验收
python .\Test_Day10.py --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k

# 正式保存图结构和审计报告
python .\run_day10.py --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k --output-dir .\experiments\day10_ml100k
```

如果 Day 7 的目录名称不同，请替换 `--run-dir`。该目录应包含：

```text
config.json
train_pairs.tsv
valid_pairs.tsv
test_pairs.tsv
```

验收 F 需要 `config.json` 中 `dataset=ml100k`、用户数 943、物品数 1682，三份交互共覆盖 100,000 条原始评分记录；未提供官方 MovieLens 数据时，不能将模拟数据的成功视为真实实验完成。

## 四、所有核心接口

| 函数 | 输入与结果 | 重点 |
|---|---|---|
| `item_global_id` | 物品 ID → 图全局编号 | 全局物品偏移量是 `num_users` |
| `build_interaction_matrix` | `(u,i)` → 稠密 `[U,I]` | 二值、重复边不累计 |
| `build_bipartite_adjacency` | `(u,i)` → 稀疏 `[U+I,U+I]` | 双向、无自环、COO、`coalesce` |
| `node_degrees` | A → `[U+I]` | 包含孤立节点，握手定理 |
| `neighbors_of` | A + 全局节点 ID → 邻居列表 | 只查询图中的原始边 |
| `verify_disjoint_splits` | train、valid、test → 数量记录 | 禁止重复或数据泄漏 |

固定示例：`U=2`、`I=3`、`train=[(0,0),(0,2),(1,0),(1,1)]`。

```text
R = [[1,0,1],
     [1,1,0]]

A = [[0,0,1,0,1],
     [0,0,1,1,0],
     [1,1,0,0,0],
     [0,1,0,0,0],
     [1,0,0,0,0]]

degree = [2,2,2,1,1]
```

## 五、实验协议说明

1. **Day 10 的图只使用 `train_pairs.tsv`**。验证/测试只能用于核对独立性，不得拼进训练图。
2. 用户 `u` 的全局 ID 就是 `u`；物品 `i` 的全局 ID 为 `num_users+i`。
3. 图的边是无向的；`A[u,U+i]` 和 `A[U+i,u]` 都应为 1，但无向边仍然只算一条。
4. 不构造同类型节点之间的边，也不额外添加自环。
5. 本日的 A 尚未归一化。`D^(-1/2) A D^(-1/2)`、邻居聚合、`torch.sparse.mm` 留给 Day 11。
6. Day 7 使用的零基编号与 RecBole 自己的 Padding Token 映射不同；不要混淆两套编号。
7. 当训练集中部分用户或物品没有交互时，仍应保留它们对应的零度节点。

## 六、学习来源

- Stanford CS224W 笔记：<https://snap-stanford.github.io/cs224w-notes/preliminaries/introduction-graph-structure>
- LightGCN 论文：<https://arxiv.org/abs/2002.02126>
- PyTorch 稀疏 COO：<https://docs.pytorch.org/docs/stable/sparse.html>

本练习包**不提供 TODO 的解答**。自动化验收脚本已使用独立参考实现通过合成测试，数据集成测试也已经使用模拟的 Day 7 格式数据验证其流程；官方 MovieLens 实验仍需本地运行。
