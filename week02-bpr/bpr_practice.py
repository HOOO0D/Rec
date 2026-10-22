"""第 6 天：BPR 论文与成对排序练习。实现六个接口，不使用 RecBole；第 7 天之前无需完整采样器或训练循环。使用 torch 并保留自动求导连接。"""
from __future__ import annotations

import torch


def is_valid_triplet(u: int, i: int, j: int,
                     positive_by_user: dict[int, set[int]],
                     num_items: int) -> bool:
    """仅当 i 是用户 u 的已观察正样本、j 不是时返回 True。u 必须存在于 positive_by_user 中；物品 ID 位于 [0, num_items)。拒绝相同或越界的 i、j；未观察到不代表不喜欢。"""
    # TODO A1: 实现三元组合法性约束。
    if u not in positive_by_user:
        return False
    positive = positive_by_user[u]
    return (i in range(num_items)and j in range(num_items)) and  ((i in positive) and(j not in positive))
    raise NotImplementedError


def enumerate_triplets(positive_by_user: dict[int, set[int]],
                       num_items: int) -> list[tuple[int, int, int]]:
    """按 (u,i,j) 排序枚举全部合法三元组，仅用于微型教学数据。复杂度为 O(sum_u |I_u+|*(I-|I_u+|))，不适合大规模负采样。无正样本或全部物品均为正样本的用户不产生三元组。"""
    # TODO A2: 枚举全部合法 (u,i,j)，不要采样。
    ans = []
    # 直接遍历字典只能返回键
    for u, S in positive_by_user.items():
        for i in S:
            for j in range(num_items):
                if is_valid_triplet(u,i,j,positive_by_user,num_items):
                    ans.append((u,i,j))
    return sorted(ans)
    raise NotImplementedError


def pairwise_scores(user_factors: torch.Tensor,
                    item_factors: torch.Tensor,
                    triplets: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回正样本得分、负样本得分和得分差。user_factors：[U,k]；item_factors：[I,k]；triplets：torch.long 类型的 [B,3]，每行为 (u,i,j)。输出均为 [B]，保留自动求导连接。今天无需 nn.Module 或 nn.Embedding。""" 
    # TODO B: 提取对应行，计算配对点积及得分差。
    user_ids = triplets[:,0]
    positive_item_ids = triplets[:,1]
    negative_item_ids = triplets[:,2]

    u = user_factors[user_ids]
    i = item_factors[positive_item_ids]
    j = item_factors[negative_item_ids]

    positive_scores = (u*i).sum(dim=1)
    negative_scores = (u*j).sum(dim=1)
    diff = positive_scores-negative_scores
    return positive_scores,negative_scores,diff
    raise NotImplementedError


def preference_probability(score_diff: torch.Tensor) -> torch.Tensor:
    """计算 sigmoid(score_diff)，保持形状和数据类型。"""
    # TODO C1.
    return torch.sigmoid(score_diff)
    raise NotImplementedError


def bpr_loss(score_diff: torch.Tensor) -> torch.Tensor:
    """返回 -log(sigmoid(score_diff)) 的标量均值。score_diff 为正负 1000 时仍须数值稳定。提示：使用稳定的 torch.nn.functional.logsigmoid。本练习不含正则化。"""
    # TODO C2.
    import torch.nn.functional as F
    return -F.logsigmoid(score_diff).mean()
    raise NotImplementedError


def bpr_objective(user_factors: torch.Tensor,
                  item_factors: torch.Tensor,
                  triplets: torch.Tensor) -> torch.Tensor:
    """组合 pairwise_scores 和 bpr_loss，无需优化器或训练循环。"""
    # TODO D: 组合前面实现的函数，保留自动求导连接。
    pos,nega,diff = pairwise_scores(user_factors,item_factors,triplets)
    BPR_loss = bpr_loss(diff)
    return BPR_loss
    raise NotImplementedError
