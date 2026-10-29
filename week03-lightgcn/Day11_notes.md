# Day 11｜LightGCN 归一化邻居聚合学习与实验记录

> 对应 RoadMap：Week 3 · Day 11。建议投入 4–5 小时。先手算，再编码，最后接 Day 10 的冻结训练图。

## 一、学习范围（建议 75 分钟）

- Stanford CS224W 的 GNN / Neighborhood Aggregation、Message Passing 小节：理解邻居消息与节点更新。
- LightGCN 原论文 Section 3.1：只关注归一化邻域聚合，不提前实现完整多层传播与层融合。
- PyTorch `torch.sparse.mm`、COO 的 `indices()`、`values()`、`coalesce()`：只要求 COO × Dense。

关键约定：LightGCN 采用无自环的用户—物品训练二部图；孤立节点的逆平方根度数设为 0。今天的特征矩阵仅用于传播验证，不是最终训练 Embedding。

## 二、必须独立手算的示例（建议 30–45 分钟）

全局编号：U0=0、U1=1、I0=2、I1=3、I2=4。
训练交互：`(U0,I0)`、`(U0,I2)`、`(U1,I0)`、`(U1,I1)`。

邻接矩阵：

```text
A = [[0,0,1,0,1],
     [0,0,1,1,0],
     [1,1,0,0,0],
     [0,1,0,0,0],
     [1,0,0,0,0]]
```

初始节点特征（按 U0、U1、I0、I1、I2 排列）：

```text
E0 = [[1,0],
      [0,2],
      [2,0],
      [0,4],
      [3,1]]
```

请写出完整计算步骤：

1. `d`：5 个节点的度向量。
2. `D`：对角度矩阵。
3. `D^(-1/2)`：对角逆平方根矩阵。
4. `A_norm = D^(-1/2) A D^(-1/2)`：写出每一个非零元素。
5. U0 的一次邻居聚合：分别写出 I0、I2 的权重和消息值。
6. 全部 5 个节点的一次聚合结果 `E1 = A_norm @ E0`。
7. `A_norm` 每一行的元素和是否必须等于 1？解释原因。

**核对答案**（手算完成后再查看）：

- `d = [2,2,2,1,1]`
- `(A_norm)[U0,I0] = 1/2`，`(A_norm)[U0,I2] = 1/sqrt(2)`
- `E1[U0] = [1+3/sqrt(2), 1/sqrt(2)] ≈ [3.1213, 0.7071]`
- `E1[U1] = [1, 2sqrt(2)] ≈ [1, 2.8284]`
- `E1[I0] = [0.5, 1]`，`E1[I1] = [0, sqrt(2)]`，`E1[I2] = [1/sqrt(2), 0]`
- 对称归一化一般不会使每一行的和等于 1。与随机游走归一化 `D^(-1)A` 不同。

## 三、代码实现（建议 120 分钟）

- [ ] `degrees_from_adjacency`：得到 [N] 度向量，包括度为 0 的节点。
- [ ] `inverse_sqrt_degrees`：安全计算 `d^(-1/2)`，对 0 度数返回 0。
- [ ] `normalize_bipartite_adjacency`：只改变非零边权，保留稀疏布局和边坐标。
- [ ] `manual_node_aggregate`：明确按邻居遍历并手工计算加权求和。
- [ ] `sparse_aggregate`：必须使用 `torch.sparse.mm`，结果形状 [N,d]。

## 四、真实图集成与日志（建议 45 分钟）

输入：Day 10 的 `adjacency.pt`、`degrees.pt`、`graph_summary.json`。

```powershell
python .\Test_Day11.py --graph-dir .\experiments\day10_ml100k
python .\run_day11.py --graph-dir .\experiments\day10_ml100k --output-dir .\experiments\day11_ml100k
```

记录：

| 项目 | 实际结果 |
|---|---|
| Day 10 图路径 | |
| 训练图来源 | |
| 节点总数 | |
| 原始无向边数量 | |
| COO 非零条目数量 | |
| 孤立节点数量 | |
| 归一化后非零条目数量 | |
| 是否出现 NaN/Inf | |
| `normalized_adjacency.pt` 路径 | |
| `inverse_sqrt_degrees.pt` 路径 | |
| `one_hop_sample.pt` 路径 | |
| `normalization_summary.json` 路径 | |

## 五、理论验收（建议 30 分钟）

1. 为什么节点度是邻接矩阵对应行的和？无向图时度数之和与边数有什么关系？
2. 对称归一化中的 `1/sqrt(d_i d_j)` 为什么同时使用两个节点的度？
3. 为什么 `D^(-1/2) A D^(-1/2)` 的行和不一定为 1？它与 `D^(-1) A` 的区别是什么？
4. LightGCN 与 Kipf–Welling 经典 GCN 在自环、逐层特征变换和激活函数上的区别是什么？
5. 为何要对度为 0 的节点设置逆平方根权重 0？传播结果是什么？
6. `torch.sparse.mm(A_norm,E)` 为什么与逐节点消息累加等价？
7. 使用稀疏图有哪些内存优势？为什么不直接构造真实图的稠密度矩阵 D？
8. Day 11 的一次传播与 Day 13 的完整 LightGCN 图传播及层融合分别是什么关系？

## 六、完成状态

- [ ] `Test_Day11.py` 在无真实图时 A–E 共 80/80。
- [ ] 集成真实 Day 10 MovieLens 训练图，A–F 为 100/100。
- [ ] 手算 A_norm 和 E1，并与运行结果一致。
- [ ] 归一化图只使用训练交互，无验证/测试边泄漏。
- [ ] 完成以上理论问答，不把 Day 13 的完整 LightGCN 实现提前加入本日范围。
- 本日最难理解的知识点：
- 需要补练的内容：
- 能否进入 Day 12：
