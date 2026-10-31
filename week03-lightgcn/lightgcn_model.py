"""Day 13：LightGCN 纯前向模型独立实现模板。

与 RoadMap(2).md 对齐：
  - Day 10：训练交互构造无向二部图。
  - Day 11：计算无自环、对称归一化的 COO 邻接矩阵。
  - Day 12：学习 LightGCN 论文第 3 节。
  - Day 13：多层传播、层融合、评分、梯度验证（本文件）。
  - Day 14：才加入 BPR Loss、负采样和完整优化器训练。

重要约定：
  - 用户 ID 使用 [0, n_users)，物品 ID 使用 [0, n_items)。
  - 邻接矩阵全局节点顺序是 [全部用户, 全部物品]。
  - Day 11 的邻接矩阵已经归一化，不能重复归一化或增加自环。
  - LightGCN 不使用可训练的 W、逐层偏置或激活函数；
    唯一可训练的参数是初始 user_embedding、item_embedding。
  - 各层 Embedding 是即时计算结果，不能在 forward 中 detach 或缓存过期值。
  - 图作为 buffer 保留，可随模型一起移动到 GPU，但不是优化对象。
"""
from __future__ import annotations

from typing import Sequence
import torch
from torch import nn


class LightGCN(nn.Module):
    """一个便于 Day 14 接入 BPR 训练的最小 LightGCN。

    参数：
        n_users：用户数量（不包含 Padding，保持 Day 7 的内部编号）。
        n_items：物品数量。
        embedding_dim：初始 Embedding 的维数 d。
        n_layers：图传播的层数 K，可以为 0。
        normalized_adjacency：Day 11 输出的归一化 COO，形状 [U+I,U+I]。
        layer_weights：可选的 K+1 个非负权重，和必须约等于 1；
            为 None 时使用均匀权重 1/(K+1)。本日不学习权重。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        n_layers: int,
        normalized_adjacency: torch.Tensor,
        layer_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        # TODO A：检查各数量为合适的整数；用户/物品数量、d 均大于 0，K >= 0。
        for value in (n_users, n_items, embedding_dim, n_layers):
            if type(value) is not int:
                raise ValueError("数量参数必须为整数")

        if n_layers < 0:
            raise ValueError("n_layers 必须大于等于 0")
        if n_items<=0 or n_users<=0 or embedding_dim <=0:
            raise ValueError()
        
        # TODO A：检查邻接矩阵为 float32 的二维 COO 方阵，形状 [U+I,U+I]。
        if normalized_adjacency.dtype != torch.float32:
            raise ValueError()
        if normalized_adjacency.layout != torch.sparse_coo:
            raise ValueError()
        if normalized_adjacency.shape != (n_users+n_items,n_users+n_items):
            raise ValueError()
        # TODO A：创建两张 nn.Embedding，并使用 normal_(mean=0,std=0.1) 初始化。
        self.user_embedding = nn.Embedding(num_embeddings=n_users,embedding_dim=embedding_dim,dtype=torch.float32)
        self.item_embedding = nn.Embedding(num_embeddings=n_items,embedding_dim=embedding_dim,dtype=torch.float32)
        nn.init.normal_(self.user_embedding.weight,mean=0,std=0.1)
        nn.init.normal_(self.item_embedding.weight,mean=0,std=0.1)
        # TODO A：调用 register_buffer 注册已 coalesce 的归一化邻接矩阵。
        self.register_buffer("normalized_adjacency",normalized_adjacency.coalesce())
        # TODO A：验证 layer_weights：长度 K+1，有限、非负，总和接近 1；
        #         将其存为不需要梯度的 buffer，不增加新的可训练参数。
        if layer_weights is None:
            weights = torch.full((n_layers+1,),1.0/(n_layers+1),dtype=torch.float32)
        else :
            weights = torch.as_tensor(layer_weights, dtype=torch.float32).detach().clone()
        if weights.ndim != 1 or weights.numel() != n_layers + 1:
            raise ValueError("层权重必须是一维张量，长度为 K+1")

        if not torch.isfinite(weights).all():
            raise ValueError("层权重不能包含 NaN 或无穷大")

        if (weights < 0).any():
            raise ValueError("层权重不能为负数")

        if not torch.isclose(
            weights.sum(),
            weights.new_tensor(1.0),
            atol=1e-6,
            rtol=1e-5,
            ):
            raise ValueError("层权重之和必须接近 1")
        self.register_buffer("layer_weights",weights)
        self.n_layers = n_layers
        return
        raise NotImplementedError("TODO A：请实现 LightGCN.__init__")

    def propagate_layers(self) -> list[torch.Tensor]:
        """返回 [E(0), E(1), ..., E(K)]，每个张量形状 [U+I,d]。

        注意：torch.sparse.mm(A_norm,E) 对稠密 Embedding 可导。
        K=0 时列表仅包含 E(0)，不执行图传播。
        """
        # TODO B：按用户、物品顺序使用 torch.cat 拼接初始 Embedding。
        embedding = torch.cat((self.user_embedding.weight,self.item_embedding.weight),dim=0)
        # TODO B：执行 K 次 torch.sparse.mm(self.normalized_adjacency,E)。
        ans = []
        ans.append(embedding)
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.normalized_adjacency,embedding)
            ans.append(E)
            embedding = E
        # TODO B：保留每一层结果；不得调用 detach、no_grad、numpy。
        return ans
        raise NotImplementedError("TODO B：请实现 propagate_layers")

    def combine_layers(self, layer_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """把 0 到 K 层按固定权重融合，输出 [U+I,d]。

        建议把 list 用 torch.stack(...,dim=0) 整理成 [K+1,U+I,d]，
        再广播权重并沿 dim=0 求和。
        """
        # TODO C：验证层数与每层形状，按照已注册权重求加权和。
        if len(layer_embeddings) != self.n_layers + 1:
            raise ValueError("层数必须为 K+1")

        expected_shape = (self.user_embedding.num_embeddings
                + self.item_embedding.num_embeddings,
                self.user_embedding.embedding_dim,)


        for embedding in layer_embeddings:
            if embedding.shape != expected_shape:
                raise ValueError("每层形状必须为 [U+I,d]")
        layer_E = torch.stack(layer_embeddings,dim = 0)
        if layer_E.shape[0] != self.n_layers+1 or layer_E.shape[1] != self.user_embedding.weight.shape[0]+self.item_embedding.weight.shape[0] or layer_E.shape[2] != self.user_embedding.weight.shape[1]:
            raise ValueError()
        # TODO C：请勿只使用最后一层，也不要无意中求和而漏掉平均。
        weights = self.layer_weights.reshape(-1,1,1)
        return (layer_E * weights).sum(dim=0)
        raise NotImplementedError("TODO C：请实现 combine_layers")

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回最终用户矩阵 [U,d] 与物品矩阵 [I,d]。"""
        # TODO D：调用 propagate_layers -> combine_layers -> split。
        layers = self.propagate_layers()
        combine_layers = self.combine_layers(layers)
        users,items = torch.split(combine_layers,[self.user_embedding.weight.shape[0],self.item_embedding.weight.shape[0]],dim=0)
        return (users,items)
        raise NotImplementedError("TODO D：请实现 forward")

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """按位置一一配对评分，返回 [B]，不是 B×B 全配对。

        例：user_ids=[0,1]，item_ids=[2,0] 返回两个分数，
        即 (U0,I2) 和 (U1,I0)；需要进行 ID 越界检查。
        """
        # TODO E：检查 ID 为 1D 的 torch.long，长度相等且范围合法。
        for name, ids in (
            ("user_ids", user_ids),
            ("item_ids", item_ids),
            ):
            if not isinstance(ids, torch.Tensor):
                raise ValueError(f"{name} 必须是 Tensor")

            if ids.ndim != 1:
                raise ValueError(f"{name} 必须是一维张量")

            if ids.dtype != torch.long:
                raise ValueError(f"{name} 必须是 torch.long")

        # 2. 按位置配对，因此两组 ID 数量必须相同
        if user_ids.numel() != item_ids.numel():
            raise ValueError("用户 ID 和物品 ID 的数量必须相同")

        # 3. 检查 ID 范围
        n_users = self.user_embedding.num_embeddings
        n_items = self.item_embedding.num_embeddings

        if ((user_ids < 0) | (user_ids >= n_users)).any():
            raise ValueError("用户 ID 越界")

        if ((item_ids < 0) | (item_ids >= n_items)).any():
            raise ValueError("物品 ID 越界")
        # TODO E：使用 forward() 生成最终 Embedding，对齐取行并按 dim=1 点积。
        user_E,item_E = self.forward()
        U = user_E[user_ids]
        I = item_E[item_ids]
        return (U*I).sum(dim=1)
        raise NotImplementedError("TODO E：请实现 score")

    def full_sort_scores(self, user_ids: torch.Tensor) -> torch.Tensor:
        """对指定 B 个用户和全部物品评分，返回 [B,n_items]。

        本日不做训练已见物品过滤；那是 Day 8 评价器的任务。
        """
        # TODO F：检查用户 ID，执行用户矩阵与物品矩阵的矩阵乘法。
        if not isinstance(user_ids, torch.Tensor):
            raise ValueError("user_ids 必须是 Tensor")

        if user_ids.ndim != 1:
            raise ValueError("user_ids 必须是一维张量")

        if user_ids.dtype != torch.long:
            raise ValueError("user_ids 必须是 torch.long")

        n_users = self.user_embedding.num_embeddings
        if ((user_ids < 0) | (user_ids >= n_users)).any():
            raise ValueError("用户 ID 越界")
        
        user_E,item_E = self.forward()
        selected_users = user_E[user_ids]
        scores = selected_users@item_E.T
        return scores
        raise NotImplementedError("TODO F：请实现 full_sort_scores")
