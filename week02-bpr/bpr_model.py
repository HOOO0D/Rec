"""第 7 天修订版：BPR-MF 模型练习模板。用户和物品 ID 从 0 开始，批量得分形状为 [B]。"""
import torch
from torch import nn


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        super().__init__()
        # TODO: 创建 user_embedding 和 item_embedding（nn.Embedding）。
        self.user_embedding = nn.Embedding(n_users,embedding_dim)
        self.item_embedding = nn.Embedding(n_items,embedding_dim)
        # TODO: 使用 nn.init.normal_(..., mean=0, std=0.1) 初始化两个权重。
        self.user_embedding.weight = nn.init.normal_(self.user_embedding.weight,0,0.1)
        self.item_embedding.weight = nn.init.normal_(self.item_embedding.weight,0,0.1)
        return
        raise NotImplementedError

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """计算配对 ID 的点积：[B] -> [B]。"""
        # TODO: 不要返回 [B,B]，批量大小为 1 时也不要压缩成标量。
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        return (u*i).sum(dim=1)
        raise NotImplementedError

    def forward(self, user_ids: torch.Tensor, pos_ids: torch.Tensor,
                neg_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回正样本得分 [B] 和负样本得分 [B]。"""
        # TODO: 复用 score()。
        pos = self.score(user_ids,pos_ids)
        neg = self.score(user_ids,neg_ids)
        return (pos,neg)
        raise NotImplementedError


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """稳定计算 softplus(-(正样本得分 - 负样本得分)) 的标量均值，不含正则化。"""
    # TODO: 保证得分差为正负 1000 时数值稳定。
    delta = neg_scores-pos_scores
    return torch.nn.functional.softplus(delta).mean()
    raise NotImplementedError
