"""Day 11｜LightGCN 邻居聚合：独立实现模板。

前置条件：已完成 Day 10，拥有未归一化的无向二部图 COO 邻接矩阵。
本日只实现 D、D^(-1/2)、D^(-1/2) A D^(-1/2) 和一层消息聚合。
不添加自环，不训练 LightGCN，不融合多层 Embedding（这些属于后续学习日）。

约定：
  - 输入邻接矩阵 A 为二维、方形、无向、无自环、非负权重的 COO 稀疏张量。
  - Day 10 的 A 为二值图，数值为 0 或 1，允许存在孤立节点。
  - 输出归一化邻接矩阵必须保持稀疏 COO 格式、float32、已合并索引。
  - 度为 0 时，把该节点的逆平方根度设为 0，避免 NaN / Inf。
  - 归一化过程中只改变已有边的权重，不增加自环或其他边。
  - 需要学习的 Embedding 是稠密 [N,d]，传播结果仍为稠密 [N,d]。
"""
from __future__ import annotations

import torch


def degrees_from_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """计算节点度，返回 [N] 的 float32 张量，包括孤立节点。

    小图使用行求和；真实图建议通过 COO 行索引和 values 计算，不转换为完整稠密矩阵。
    输入不是稀疏 COO 方阵或出现负边权、非有限边权时，抛出 ValueError。
    """
    # TODO A：检查输入的 layout 和形状，将 COO 索引合并。
    if adjacency.layout!= torch.sparse_coo:
        raise ValueError()
    adjacency = adjacency.coalesce()
    values = adjacency.values()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError()
    for v in adjacency.values():
        if v<0 or v == torch.inf or v== -torch.inf:
            raise ValueError()
    
    size = adjacency.shape[0]
    # TODO A：按行累加边权，确保孤立节点对应的度仍为 0。
    weight = torch.zeros(size=(size,),dtype=torch.float32)
    for (i,j),value in zip(adjacency.indices().T,adjacency.values()):
        weight[i] += value
    return weight
    raise NotImplementedError("请独立完成 degrees_from_adjacency")


def inverse_sqrt_degrees(degrees: torch.Tensor) -> torch.Tensor:
    """对每个节点计算 d^(-1/2)，度为零的位置必须输出零。

    输入为一维、非负且有限的度向量；非法输入抛 ValueError。
    例如 [2, 1, 0] → [1/sqrt(2), 1, 0]。
    """
    # TODO B：先判断哪些节点度数大于 0，再对这些节点求逆平方根。
    if degrees.ndim != 1:
        raise ValueError()
    if not torch.isfinite(degrees).all() or (degrees<0).any():
        raise ValueError()
    
    # TODO B：不要对零度数直接使用 pow(-0.5) 而不处理无穷大。
    result = torch.zeros_like(degrees, dtype=torch.float32)
    positive = degrees > 0
    result[positive] = degrees[positive].float().pow(-0.5)
    return result
    raise NotImplementedError("请独立完成 inverse_sqrt_degrees")


def normalize_bipartite_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """计算对称归一化邻接矩阵 D^(-1/2) A D^(-1/2)。

    输出保持 COO 稀疏且已 coalesce，形状和非零坐标与输入一致。
    Day 10 的二值无向图每条边 (i,j) 的新权重应为 1/sqrt(d_i*d_j)。
    不要调用 to_dense() 计算真实图上的完整 D 或归一化矩阵。
    """
    # TODO C：调用 degrees_from_adjacency 和 inverse_sqrt_degrees。
    D = degrees_from_adjacency(adjacency=adjacency)
    D_frac = inverse_sqrt_degrees(D)
    # TODO C：从 COO 索引读出边两端节点编号，只更新 values 的权重。
    adjacency.coalesce()
    row,col = adjacency.indices()
    values = adjacency.values()
    # TODO C：用原坐标、新 values 创建新的稀疏 COO 张量并 coalesce。
    new_values = D_frac[row]*values*D_frac[col]
    result = torch.sparse_coo_tensor(adjacency.indices(),new_values,size=adjacency.shape,dtype=torch.float32,device=adjacency.device).coalesce()
    return result
    raise NotImplementedError("请独立完成 normalize_bipartite_adjacency")


def manual_node_aggregate(
    normalized_adjacency: torch.Tensor,
    embeddings: torch.Tensor,
    node_id: int,
) -> torch.Tensor:
    """显式遍历当前节点的邻边，手工加权求和，返回形状 [d] 的向量。

    这是学习消息传递机制的小图练习，可以通过 Python 循环实现。
    不允许使用 torch.sparse.mm 代替此函数的逐边累加；
    对孤立节点返回零向量；节点越界抛出 ValueError。
    """
    if node_id < 0 or node_id >= normalized_adjacency.shape[0]:
        raise ValueError("节点编号越界")
    # TODO D：查找 normalized_adjacency 中以 node_id 为行号的所有边。
    normalized_adjacency = normalized_adjacency.coalesce()
    row, col = normalized_adjacency.indices()
    values = normalized_adjacency.values()
    result = embeddings.new_zeros(embeddings.shape[1])
    for k in range(values.numel()):
        if row[k] == node_id:
            neighbor_id = col[k]
            result += values[k] * embeddings[neighbor_id]

    return result
    # TODO D：对每条边计算 边权 × 邻居 Embedding，累加成长度为 d 的向量。
    raise NotImplementedError("请独立完成 manual_node_aggregate")


def sparse_aggregate(normalized_adjacency: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """使用 torch.sparse.mm 完成一次全图消息传播，返回 [N,d]。

    必须检查维度，不改变输入，并保持对 embeddings 的自动求导能力。
    输入不合法（维数、形状不匹配、不是 COO 稀疏图）时抛 ValueError。
    """
    if normalized_adjacency.layout != torch.sparse_coo:
            raise ValueError()
    if normalized_adjacency.shape[0]!=embeddings.shape[0]:
        raise ValueError()
    
    if normalized_adjacency.ndim != 2 or embeddings.ndim != 2 :
        raise ValueError()
    # TODO E：用 torch.sparse.mm(normalized_adjacency, embeddings) 实现。
    return torch.sparse.mm(normalized_adjacency,embeddings)
    # TODO E：不要在这里再次归一化，也不要增加自环、ReLU 或线性层。
    raise NotImplementedError("请独立完成 sparse_aggregate")
