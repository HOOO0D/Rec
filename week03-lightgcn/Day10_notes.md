# Day 10｜图论基础与用户—物品二部图

> Week 3 · Day 1（预计 4–5 小时）。今天只要求正确构造图，不实现归一化、消息传递和 LightGCN 训练。

## 一、阅读范围（建议 75 分钟）

### 必读

1. Stanford CS224W [Introduction and Graph Structure](https://snap-stanford.github.io/cs224w-notes/preliminaries/introduction-graph-structure)：只读 **Basic Concepts、Bipartite Graphs、Representing Graphs**，弄清有向/无向边、节点度、邻接矩阵和 Edge List。
2. LightGCN [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)：只读 Abstract、Introduction，理解“用户—物品交互构成图”的研究动机；不要求提前掌握完整传播公式。
3. PyTorch [sparse_coo_tensor](https://docs.pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html) 与 [Sparse COO](https://docs.pytorch.org/docs/stable/sparse.html)：只学 COO 的 `indices`、`values`、`size`、`coalesce()`、`to_dense()`。

### 今天不学

- 不学 `D^(-1/2) A D^(-1/2)` 的具体计算（Day 11）。
- 不学多层邻居聚合或 `torch.sparse.mm()`（Day 11–13）。
- 不训练 LightGCN，不计算 Recall@K，不做 RecBole 模型对照（Day 14–15）。
- 不强制学习 NetworkX、PyG、DGL、CS224W 全部课程。

## 二、手算任务（建议 30 分钟）

训练集只有四条交互：

```text
(user_id, item_id)
(0,0)
(0,2)
(1,0)
(1,1)
```

用户数量 U=2，物品数量 I=3，所有编号从 0 开始。

1. 分别画出用户节点 U0、U1 和物品节点 I0、I1、I2，标出 4 条训练边。
2. 写出 R，形状是几行几列？
3. 采用全局节点顺序 `[U0, U1, I0, I1, I2]`，列出每个节点的整数编号。
4. 写出 `A` 的全部 25 个元素，形状是什么？
5. 验证 `A == A.T`，并说明为何左上角与右下角的两个对角块都是 0。
6. 逐个计算 5 个节点的度数，并验证 `sum(degree)=2×|E|`。
7. 计算 U0、I0 的邻居集合，以及 U0 经过两条边能到达的其他用户（不计算传播）。
8. 假设验证集 `(0,1)`、测试集 `(1,2)`，指出它们在 **训练邻接矩阵** 的对应两个位置，并说明为什么必须为 0。

### 重要提示

- 交互矩阵 `R` 是 `[U,I]`，图邻接矩阵 `A` 是 `[U+I,U+I]`。
- 无向边在邻接矩阵中表现为两个对称非零位置，但图中的无向边只算 1 条。
- 节点度为邻接矩阵每一行的元素和；握手定理：`sum(degree)=2×|E|`。
- 不要为了迎合模型结构而添加自环；本日采用零对角线的二值二部图。

## 三、代码实现（建议 120 分钟）

在 `graph_basics.py` 中独立完成六个函数：

- [ ] `item_global_id`：物品局部 ID → 全局节点编号；越界抛 ValueError。
- [ ] `build_interaction_matrix`：生成二值 `[U,I]` 矩阵。
- [ ] `build_bipartite_adjacency`：生成对称、无自环、二值 `COO` 稀疏邻接矩阵。
- [ ] `node_degrees`：包含孤立节点的完整度向量。
- [ ] `neighbors_of`：按编号升序列出某节点的邻居。
- [ ] `verify_disjoint_splits`：拒绝重复或交叉划分，保留数据审计能力。

要求不用 RecBole、NetworkX、PyG 直接完成核心逻辑；可以使用 PyTorch 张量和 Python 标准库。

## 四、自动验收（建议 30 分钟）

| 测试 | 内容 | 分数 |
|---|---|---:|
| A | 全局节点编号和合法性 | 10 |
| B | 二值交互矩阵 R | 15 |
| C | 对称 COO 稀疏邻接矩阵 A | 25 |
| D | 节点度、邻居、握手定理 | 15 |
| E | 重复交互、孤立节点、划分无泄漏 | 15 |
| F | 真实 MovieLens-100K 训练交互构图和保存 | 20 |
| **合计** | **六项全部通过** | **100** |

```powershell
cd E:\AI\rec-learning\week03-lightgcn
python .\Test_Day10.py
python .\Test_Day10.py --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k
```

仅 A–E 通过是 80/100，需正式执行 F 才算代码验收完成。完整的学习验收还包括独立手算。

## 五、MovieLens 集成实验与输出（建议 45 分钟）

从 Day 7 已冻结的训练交互构建：

```powershell
python .\run_day10.py --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k --output-dir .\experiments\day10_ml100k
```

期望：

- `num_users=943`、`num_items=1682`、总节点数 `2625`（Day 7 自己的零基 ID，不是 RecBole 的 padding ID）。
- 无向边数 = **训练集去重交互条数**，存储的双向条目数 = 2 × 无向边数。
- 保存 `adjacency.pt`、`degrees.pt`、`graph_summary.json`。
- `graph_source=train_only`；验证集和测试集仅用于检查交互不重叠，不能用于建边。
- 不要求所有节点的度非零；训练中未出现的物品仍保留为孤立节点。

## 六、理论问答与复盘（建议 30 分钟）

1. 图、无向图和二部图分别是什么？
2. 为什么推荐系统的用户—物品图不能直接把 user ID 和 item ID 使用同一编号空间？
3. 为什么二部图的邻接矩阵可以写成 `[[0,R],[R.T,0]]`？
4. 交互只有 4 条，为什么 COO 邻接矩阵中有 8 个非零元素？
5. 邻接矩阵是对称的，是否说明它是有向图？（解释真正的语义。）
6. 如果同一 `(u,i)` 在 TSV 出现两次，为什么今天仍然只保留一条二值边？
7. 为什么 MovieLens 的这张图适合稀疏存储，不能只考虑稠密矩阵？
8. 为什么训练图不能包含验证/测试正交互？为什么这会影响未来的 LightGCN 实验？
9. 与 Day 7 的 BPR-MF 相比，图结构提供了什么额外的建模途径？（不要求推导传播。）
10. 今天哪些内容在 Day 11、Day 13、Day 14 会被直接复用？

### 实验记录模板

- Python / PyTorch 版本：
- Day 7 输入目录：
- 训练/验证/测试交互数量：
- 用户节点数量 / 物品节点数量 / 总节点数量：
- 无向边数 / COO 非零条目数：
- 孤立节点数：
- `train_pairs_sha256`：
- A–E 得分：
- F 得分：
- `graph_summary.json` 路径：
- 稀疏图保存文件：
- 手算是否独立完成：
- Day 11 需要继续学习的问题：
