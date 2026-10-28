"""Day 10：用户—物品二部图基础（待独立实现）。

约定：
    训练交互采用零基用户和物品编号：(u, i)。
    用户节点全局编号为 [0, num_users)。
    物品节点全局编号为 [num_users, num_users + num_items)。
    每条训练交互对应一条无向边，在邻接矩阵中存储两个对称位置。
    本日不添加自环、不对边加权、不计算归一化邻接矩阵。
    同一训练交互即使重复提供，也只视为一条二值边。
    验证集与测试集绝对不能用于构造图结构。

提示：PyTorch 2.5 支持 torch.sparse_coo_tensor(...).coalesce()。
"""

from __future__ import annotations

import torch


def item_global_id(item_id: int, num_users: int, num_items: int) -> int:
    """将物品内部编号转换为图中的全局节点编号。

    例：num_users=2，num_items=3 时，I0 -> 2、I1 -> 3、I2 -> 4。
    如果物品编号越界，或用户/物品数量为负数，应抛出 ValueError。
    """
    # TODO A：正确实现节点编号映射与参数边界检查。
    if num_users<0 or num_items<0:
        raise ValueError()
    if item_id< 0:
        raise ValueError()
    item_ID = num_users+item_id
    if item_ID< num_users+num_items:
        return item_ID
    else:
        raise ValueError()
    raise NotImplementedError("请完成 item_global_id")


def build_interaction_matrix(
    train_pairs: list[tuple[int, int]],
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    """构造二值交互矩阵 R，返回 float32 的稠密张量，形状 [U, I]。

    训练交互 (u, i) 的位置赋值为 1；其他位置为 0。
    相同 (u, i) 重复出现时结果仍为 1，不累加为 2。
    如果用户/物品编号越界或节点数量为负数，应抛 ValueError。
    """
    # TODO B：先检查输入范围，再根据训练交互填充矩阵。
    if num_items<0 or num_users<0:
        raise ValueError()
    adj = torch.zeros(size = (num_users,num_items),dtype=torch.float32)
    # TODO B：不能假设原始用户编号和物品编号来自同一编号空间。
    for u,i in train_pairs:
        u_id = u
        i_id = i
        if u_id>=num_users or i_id >=num_items:
            raise ValueError()
        adj[u_id,i_id]=1
    return adj
    raise NotImplementedError("请完成 build_interaction_matrix")


def build_bipartite_adjacency(
    train_pairs: list[tuple[int, int]],
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    """构造无向二部图的稀疏 COO 邻接矩阵 A，形状 [U+I, U+I]。

    要求：
      1. 返回布局 torch.sparse_coo，且调用 coalesce() 合并重复坐标。
      2. 对每条去重交互 (u,i) 添加 (u,U+i) 和 (U+i,u) 两个对称元素。
      3. 所有非零边权恰好为 1.0（float32），不是交互次数。
      4. 不出现用户—用户边、物品—物品边、自环。
      5. 没有训练交互时返回指定大小的全零稀疏邻接矩阵。

    提示：用 indices=[行索引列表, 列索引列表] 与 values=全 1 向量构造。
    """
    # TODO C：建议先用集合去重训练交互，再构造双向索引和 COO 张量。
    train_pairs = set(train_pairs)
    rows = []
    cols = []
    for u,i in train_pairs:
        if u<0 or u>=num_users:
            raise ValueError()
        u_id = u
        i_id = item_global_id(i,num_users,num_items)
        rows.append(u_id)
        rows.append(i_id)
        cols.append(i_id)
        cols.append(u_id)
    indices = torch.tensor([rows,cols],dtype=torch.int64)
    values = torch.ones(len(rows),dtype=torch.float32)
    return torch.sparse_coo_tensor(indices=indices,values=values,size=(num_users+num_items,num_items+num_users)).coalesce()
    raise NotImplementedError("请完成 build_bipartite_adjacency")


def node_degrees(adjacency: torch.Tensor) -> torch.Tensor:
    """返回所有全局节点的度数，形状 [U+I]，类型 torch.int64。

    传入的是未归一化、二值、无向图的稀疏 COO 邻接矩阵。
    孤立节点的度必须为 0，不允许被排除。
    """
    # TODO D：对邻接矩阵按行计数，注意不要把双向边错误地多算两遍。
    indices = adjacency.indices()
    size = adjacency.size()[0]
    node = torch.zeros(size=[size,],dtype=torch.int64)
    for i in indices[0]:
        node[i]+=1
    return node
    raise NotImplementedError("请完成 node_degrees")


def neighbors_of(adjacency: torch.Tensor, node_id: int) -> list[int]:
    """查询给定全局节点的所有邻居，按节点编号升序返回。

    输入节点不存在时应抛 ValueError；孤立节点应返回 []。
    """
    # TODO D：检查编号，读取 COO 的行列坐标并返回该节点邻居。
    if node_id<0 or node_id>=adjacency.size()[0]:
        raise ValueError()
    indices = adjacency.indices()
    nei = []
    for i,j in indices.t():
        if i == node_id:
            nei.append(j)
    return nei
    raise NotImplementedError("请完成 neighbors_of")


def verify_disjoint_splits(
    train_pairs: list[tuple[int, int]],
    valid_pairs: list[tuple[int, int]],
    test_pairs: list[tuple[int, int]],
) -> dict[str, int]:
    """审计三份交互划分，保证独立性并返回各自原始记录数量。

    不能修改输入数据。任何集合内部重复、跨集合交叉、负数 ID
    或非整数 ID，均需抛出 ValueError。
    返回键必须为 n_train、n_valid、n_test。
    """
    # TODO E：此函数只用于审核划分，不负责把验证/测试边加入图。
    pair_sets = []
    for pairs in (train_pairs,valid_pairs,test_pairs):
        for u,i in pairs:
            if type(u)is not int or type(i)is not int:
                raise ValueError()
            if u<0 or i<0:
                raise ValueError()
        pair_set = set(pairs)
        if len(pair_set) != len(pairs):
            raise ValueError()
        pair_sets.append(pair_set)
    # 序列解包赋值
    train_set, valid_set, test_set = pair_sets
    if train_set & valid_set:
        raise ValueError("训练集与验证集存在交叉")
    if train_set & test_set:
        raise ValueError("训练集与测试集存在交叉")
    if valid_set & test_set:
        raise ValueError("验证集与测试集存在交叉")

    return {
        "n_train": len(train_pairs),
        "n_valid": len(valid_pairs),
        "n_test": len(test_pairs),
    }
    raise NotImplementedError("请完成 verify_disjoint_splits")
