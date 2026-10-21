"""Day 6 | BPR paper and pairwise ranking practice (TODO only).

Implement six interfaces without using RecBole. No complete sampler or model
training is required until Day 7. Use torch and preserve autograd gradients.
"""
from __future__ import annotations

import torch


def is_valid_triplet(u: int, i: int, j: int,
                     positive_by_user: dict[int, set[int]],
                     num_items: int) -> bool:
    """True iff i is an observed positive of u and j is not.

    Requirements: u must be present in positive_by_user; item IDs must lie in
    [0, num_items). Reject identical i/j and out-of-range IDs. Do not assume
    an unobserved j is an actual disliked item.
    """
    # TODO A1: implement the triplet constraints.
    if u not in positive_by_user:
        return False
    positive = positive_by_user[u]
    return (i in range(num_items)and j in range(num_items)) and  ((i in positive) and(j not in positive))
    raise NotImplementedError


def enumerate_triplets(positive_by_user: dict[int, set[int]],
                       num_items: int) -> list[tuple[int, int, int]]:
    """Enumerate ALL eligible triples in sorted (u, i, j) order.

    Educational use on tiny data only! This O(sum_u |I_u+|*(I-|I_u+|))
    construction is not an efficient large-scale negative sampler.
    Users with zero positives or with all items positive give no triples.
    """
    # TODO A2: enumerate, do not sample, all legal (u,i,j).
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
    """Return (positive_scores, negative_scores, score_differences).

    user_factors: [U, k], item_factors: [I, k]
    triplets: torch.long [B,3] of (u,i,j)
    each output: [B] and remains connected to autograd.
    No `nn.Module` or `nn.Embedding` is necessary today.
    """ 
    # TODO B: gather rows, dot products for same user, score difference.
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
    """Compute sigmoid(score_diff), preserving its shape and dtype."""
    # TODO C1.
    return torch.sigmoid(score_diff)
    raise NotImplementedError


def bpr_loss(score_diff: torch.Tensor) -> torch.Tensor:
    """Return mean -log(sigmoid(score_diff)) as a scalar tensor.

    Must be numerically stable even for score_diff = -1000 or +1000.
    Hint: torch.nn.functional.logsigmoid is stable.
    IMPORTANT: no regularization here, to keep this Day 6 exercise focused.
    """
    # TODO C2.
    import torch.nn.functional as F
    return -F.logsigmoid(score_diff).mean()
    raise NotImplementedError


def bpr_objective(user_factors: torch.Tensor,
                  item_factors: torch.Tensor,
                  triplets: torch.Tensor) -> torch.Tensor:
    """Compose pairwise_scores and bpr_loss; no optimizer/train loop."""
    # TODO D: compose your previous functions. Preserve autograd.
    pos,nega,diff = pairwise_scores(user_factors,item_factors,triplets)
    BPR_loss = bpr_loss(diff)
    return BPR_loss
    raise NotImplementedError
