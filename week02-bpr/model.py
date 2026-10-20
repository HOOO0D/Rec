"""Day 5：矩阵分解（MF）自主练习模板。

使用：将本文件与 Test_Day5.py 放在同一目录，建议重命名为 model.py。
只填充 TODO，勿修改公开接口或自动验收脚本。

定义：
  forward(user_ids, item_ids) -> [B]  成对评分
  score_all() -> [n_users, n_items]  所有用户对所有物品的评分
  l2_regularization() -> scalar  两张完整 Embedding 表的平方和
  mf_loss(...) -> scalar  仅观测样本上的 MSE + reg_lambda * L2
  train_mf(...) -> list[float]  固定给定样本的全批量训练 Loss 记录

范围：无 bias 的基础 MF。用户偏置、物品偏置作为理论理解，不参与今日测试。
"""

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        super().__init__()
        # TODO A1: 定义两个可训练 nn.Embedding：
        # self.user_embedding： [n_users, embedding_dim]
        # self.item_embedding： [n_items, embedding_dim]
        self.user_embedding = nn.Embedding(num_embeddings=n_users,embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=n_items,embedding_dim=embedding_dim)
        return
        raise NotImplementedError("TODO A1: 实现模型初始化")

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """输入两个形状 [B] 的 LongTensor，返回成对点积评分 [B]。"""
        # TODO A2: 先查表，再对每对向量求点积；B=1 时仍返回 [1]
        # Embedding 得到的是[B,k]
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        return (u*i).sum(dim=1)
        raise NotImplementedError("TODO A2: 实现批量评分")

    def score_all(self) -> torch.Tensor:
        """返回全量预测评分矩阵，形状 [n_users, n_items]。"""
        # TODO B: 使用两个 Embedding 的 weight 做矩阵乘法
        return self.user_embedding.weight @ self.item_embedding.weight.T
        raise NotImplementedError("TODO B: 实现全量评分")

    def l2_regularization(self) -> torch.Tensor:
        """返回 ||P||_F^2 + ||Q||_F^2，需保留梯度。"""
        # TODO C: 直接使用两个完整 weight；是“平方和”，不是开根号后的范数
        return (self.user_embedding.weight **2).sum()+(self.item_embedding.weight**2).sum()
        raise NotImplementedError("TODO C: 实现 L2 正则项")


def mf_loss(
    model: MatrixFactorization,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    ratings: torch.Tensor,
    reg_lambda: float = 0.0,
) -> torch.Tensor:
    """只在输入的已观测 (user,item,rating) 上计算平均平方误差，加 L2 惩罚。"""
    # TODO D: 要求返回 scalar Tensor。不要将未观察评分填零再计算 MSE。
    r = model(user_ids,item_ids)
    L2 = model.l2_regularization()
    N = ratings.numel()
    Loss = 1.0/N*(((ratings-r)**2).sum())+reg_lambda*L2
    return Loss
    raise NotImplementedError("TODO D: 实现 MF 的训练损失")


def train_mf(
    model: MatrixFactorization,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    ratings: torch.Tensor,
    epochs: int = 300,
    lr: float = 0.05,
    reg_lambda: float = 0.0,
) -> list[float]:
    """全批量训练。每个 epoch 记录一次更新前的 objective，返回长度为 epochs 的列表。"""
    # TODO E: Adam、zero_grad、mf_loss、backward、step。记录每轮更新前 loss.item()。
    # 此练习沿用 Day3 的训练闭环，今天重点是 MF 表示与正则化。
    Optim = torch.optim.Adam(model.parameters(),lr)
    Loss = []
    for epoch in range(epochs):
        Optim.zero_grad()
        L = mf_loss(model,user_ids,item_ids,ratings,reg_lambda)
        L.backward()
        Optim.step()
        Loss.append(L.item())
    return Loss    
    raise NotImplementedError("TODO E: 实现训练和 Loss 记录")
