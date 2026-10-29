# Day 11｜归一化邻居聚合与 `torch.sparse.mm`

本日直接承接 `RoadMap.md` 的 Day 11 和 Day 10 生成的未归一化、无自环的稀疏二部图。核心验收是手算 `D^(-1/2) A D^(-1/2)`、完成一次消息聚合、检查孤立节点无 NaN/Inf。

## 文件清单

| 文件 | 用途 |
|---|---|
| `graph_aggregation.py` | 5 个核心 TODO：节点度、安全逆平方根、归一化、逐节点聚合、稀疏矩阵乘法 |
| `Test_Day11.py` | 6 组独立验收，80 分合成测试 + 20 分 Day 10 MovieLens 图集成 |
| `run_day11.py` | 读取 Day 10 的图文件，验证图源，保存归一化图和传播检查结果 |
| `Day11_notes.md` | 公式、手算题、理论问答、实验记录 |
| `README_Day11.md` | 安装和运行说明 |

**中文注释要求**：所有练习源码的注释、TODO 和文档字符串均使用中文。

## 放置目录

```text
E:\AI\rec-learning\week03-lightgcn\
├── graph_basics.py                       # Day 10 已完成
├── Test_Day10.py
├── run_day10.py
├── graph_aggregation.py                  # Day 11 待完成
├── Test_Day11.py                         # Day 11 验收
├── run_day11.py                          # Day 11 产物生成
├── Day11_notes.md
└── experiments\
    ├── day10_ml100k\
    │   ├── adjacency.pt
    │   ├── degrees.pt
    │   └── graph_summary.json
    └── day11_ml100k\                    # 本日运行后生成
        ├── normalized_adjacency.pt
        ├── inverse_sqrt_degrees.pt
        ├── one_hop_sample.pt
        └── normalization_summary.json
```

## 运行方式（Windows PowerShell）

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week03-lightgcn

# 先独立完成 graph_aggregation.py 的五个 TODO。
# A–E：合成数据练习，最多 80 分。
python .\Test_Day11.py

# F：依赖 Day 10 的真实 MovieLens 图，完整 100 分。
python .\Test_Day11.py --graph-dir .\experiments\day10_ml100k

# 正式生成可供 Day 13 使用的归一化图。
python .\run_day11.py --graph-dir .\experiments\day10_ml100k --output-dir .\experiments\day11_ml100k
```

如果 Day 10 的图保存在别处，修改 `--graph-dir`，不必重新构造或随机划分数据。

## 必须遵守的数学与工程协议

1. 输入是 **Day 10 原始无向二部图**，含用户与物品所有全局节点，不添加自环。
2. 度向量 `d_i = sum_j A[i,j]`；度矩阵 D 只需在纸上写出，代码中不要真的构造 N×N 稠密 D。
3. `A_norm[i,j] = A[i,j] / sqrt(d_i*d_j)`；孤立节点的逆平方根度定义为 0。
4. COO 输出必须 `coalesce()`，保持原边坐标与非零数量，保证归一化权重有限。
5. `torch.sparse.mm(A_norm, E0)` 输出稠密 `[N,d]`，与小图上的 `A_norm.to_dense() @ E0` 数值一致。
6. 本日只计算一次邻域传播，不训练 Embedding、不融合多层、不计算 BPR Loss。完整实现安排在 Day 13–14。
7. 图来源必须是 `train_only`；不可把验证/测试交互添加到图中。

## 测试分值

| 测试 | 分值 |
|---|---:|
| A. 节点度与安全逆平方根 | 15 |
| B. 对称归一化边权与稀疏格式 | 25 |
| C. 逐节点手算消息聚合 | 15 |
| D. 稀疏乘法与梯度验证 | 15 |
| E. 孤立节点与空图边界 | 10 |
| F. Day 10 MovieLens 图集成 | 20 |
| **合计** | **100** |

没有 `--graph-dir` 时测试只能得到 80/100。F 只检查输入图文件格式、摘要来源和归一化结果是否相符；若要核实图确实来自官方 MovieLens-100K，仍需追溯 Day 7 数据来源及 Day 10 的冻结交互哈希。实验日志不能只凭文件名认定数据集真实性。

## 学习资料

- RoadMap Day 11：图归一化与单步邻居聚合。
- Stanford CS224W：https://snap-stanford.github.io/cs224w-notes/machine-learning-with-networks/graph-neural-networks
- LightGCN：https://hexiangnan.github.io/papers/sigir20-LightGCN.pdf
- PyTorch `torch.sparse.mm`：https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html
- PyTorch Sparse COO：https://docs.pytorch.org/docs/stable/sparse.html
