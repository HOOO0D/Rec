# Day 13｜LightGCN 多层传播、层融合与评分

## 0. 今日定位

今天接续 Day 10 的用户—物品二部图与 Day 11 的归一化邻接矩阵；Day 12 已阅读论文；今天 **只实现 LightGCN 最小前向与参数梯度验证**。BPR 训练及评价安排在 Day 14，层数消融在 Day 15。

## 1. 核心公式与边界

- `U`：用户数；`I`：物品数；`d`：Embedding 维数；`K`：传播层数。
- 全局图节点顺序：`U0 ... U(U-1) I0 ... I(I-1)`。
- 用户向量矩阵 `P0` 为 `[U,d]`；物品向量矩阵 `Q0` 为 `[I,d]`。
- `E0 = concat(P0,Q0)` 为 `[U+I,d]`。
- 对称归一化图 `A_norm` 为 `[U+I,U+I]` 的稀疏 COO（无自环）。
- 传播：`E[k+1] = sparse.mm(A_norm,E[k])`。
- 层融合：`E_final = sum(alpha[k]*E[k] for k=0..K)`。默认 `alpha[k]=1/(K+1)`。
- 评分：`score(u,i) = (U_final[u]*I_final[i]).sum()`；全排序评分：`U_final @ I_final.T`。
- 可训练参数只有第 0 层的两张 Embedding；图及固定层权重使用 buffer。

## 2. 手算题（必须独立计算）

2 用户、3 物品，节点顺序 `U0 U1 I0 I1 I2`：

\[
\widetilde A=\begin{bmatrix}
0&0&1/2&0&1/\sqrt2\\
0&0&1/2&1/\sqrt2&0\\
1/2&1/2&0&0&0\\
0&1/\sqrt2&0&0&0\\
1/\sqrt2&0&0&0&0
\end{bmatrix}
\]

\[
E^{(0)}=\begin{bmatrix}
1&0\\0&2\\2&0\\0&4\\3&1
\end{bmatrix}
\]

已知 Day 11 的一次传播：

\[
E^{(1)}=\begin{bmatrix}
1+3/\sqrt2&1/\sqrt2\\
1&4/\sqrt2\\
1/2&1\\
0&2\sqrt2\\
1/\sqrt2&0
\end{bmatrix}
\]

### TODO：

1. 逐行计算 `E(2)=A_norm @ E(1)`。
2. 求 `E_final=(E0+E1+E2)/3`。
3. 计算 `(U0,I0)` 与 `(U0,I1)` 的点积评分。
4. 解释为什么 `K=0` 时，与 MF 评分完全一致。
5. 解释为什么 `K=2` 时用户侧有两跳同侧协同信息，但用户也可能通过往返路径重新收到自己的信息。

**自查：** `E(2)` 的 U0 行应为 `[0.75, 0.5]`；这是用于定位问题的检查点，不代表全部答案。

## 3. 模型接口与形状

| 调用 | 输入 | 输出 |
|---|---|---|
| `propagate_layers()` | 模型内部的图、两张 Embedding 表 | 共 K+1 个 `[U+I,d]` |
| `combine_layers(layers)` | K+1 个 `[U+I,d]` | `[U+I,d]` |
| `forward()` | 无 | `([U,d],[I,d])` |
| `score(user_ids,item_ids)` | 两个 `[B]` 的 long Tensor | `[B]` |
| `full_sort_scores(user_ids)` | `[B]` 的 long Tensor | `[B,I]` |

## 4. 梯度与实现原则

- 不可用 `detach()` / `torch.no_grad()` 包围训练时需要求导的传播。
- 不需要为第 1 层、第 2 层构造独立的 Embedding 表。
- 正向传播不使用 `relu`、独立 `W`、每层 `bias`、额外自环。
- 不能把 `E_final` 永久缓存成与初始参数脱离的张量；Day 14 训练时每次更新后均应重新传播。
- Day 11 已经归一化，不要重复执行 `D^(-1/2)AD^(-1/2)`。
- `full_sort_scores()` 不过滤训练已见物品；调用 Day 8 的评价器时才过滤。
- 本日不要求 BPR Loss 下降；只有 Day 14 接入训练后才有意义。

## 5. 实验记录模板

- 日期：
- PyTorch 版本：
- Python/Conda 环境：
- 图来源（Day 11 normalization_summary.json）：
- `n_users, n_items, d, K`：
- `nnz(A_norm)`：
- K=0 MF 等价测试结果：
- K=1/2 手算与代码误差：
- 参数数量与可训练参数清单：
- 梯度范数（User Embedding）：
- 梯度范数（Item Embedding）：
- 孤立节点与空图：
- Day 13 自动验收：
- Day 14 待解决问题：

## 6. 理论自测

1. 为什么 LightGCN 的每一层输出在训练中可导，但又不是独立 `nn.Parameter`？
2. 为什么 K=0 与 MF 等价，但不表示 K>0 的 LightGCN 预测一定优于 MF？
3. 层融合的权重如果均匀设置，有几个权重？为什么不是 K 个？
4. 当某个节点孤立时，传播层结果为 0，为什么最终融合表示仍可以不为 0？
5. `score()` 与 `full_sort_scores()` 的 Shape 分别是什么？
6. 为什么不能把评分矩阵缓存起来再使用于反向传播与参数更新？
7. 为什么 Day 11 的 graph_source 必须是 train_only？
