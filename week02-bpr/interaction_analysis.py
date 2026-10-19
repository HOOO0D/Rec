"""Day 4｜用户—物品交互数据分析（练习模板）。

只使用 PyTorch 基础张量操作，不实现完整 UserCF、ItemCF 或 MF。
补全下面 6 个函数，然后运行同目录中的 Test_Day4.py。
"""

import torch


def build_explicit_matrix(n_users: int, n_items: int, records: list[tuple[int, int, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """构建显式评分矩阵及观察掩码。

    Args:
        records: [(user_id, item_id, rating), ...]，用户/物品 ID 从 0 开始，
                 同一 (u, i) 在此练习中至多出现一次。
    Returns:
        ratings: float32 [U, I]；未观察位置使用 NaN，评分 0 是有效评分。
        observed: bool [U, I]；True 表示该位置有观测记录。
    """
    # TODO: 不得将未观察交互填为 0，因为 0 可能是真实评分。
    scores = torch.full((n_users,n_items),float('nan'))
    observed = torch.full((n_users,n_items),bool(False))
    for u,i,r in records:
        scores[u,i] = r
        observed[u,i] = True
    return (scores,observed)
    raise NotImplementedError


def build_implicit_matrix(n_users: int, n_items: int, events: list[tuple[int, int]]) -> torch.Tensor:
    """将已观察到的隐式交互映射为 0/1 张量，形状 [U,I]。

    1 表示有记录的交互；0 仅表示没有记录到交互，不能解释为不喜欢。
    同一用户—物品即使出现多次，也记为 1。
    """
    # TODO
    records = torch.full((n_users,n_items),0)
    for u,i in events:
        records[u,i] = 1
    return records    
    raise NotImplementedError


def user_overlap(interactions: torch.Tensor, u: int, v: int) -> int:
    """用户 u 与 v 共同交互过的物品数量。interactions 形状 [U,I]。"""
    # TODO: 使用逻辑与计算交集，返回 Python int。
    return (interactions[u,:] & interactions[v,:]).sum(dim=-1).item()
    raise NotImplementedError


def item_overlap(interactions: torch.Tensor, i: int, j: int) -> int:
    """物品 i 与 j 被同一用户共同交互的用户数量。"""
    # TODO: 注意矩阵哪一维代表物品。
    return (interactions[:,i]&interactions[:,j]).sum(dim=-1).item()
    raise NotImplementedError


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """两个长度相同的 1D 0/1 向量的 Jaccard 相似度。

    J(A,B) = |A∩B| / |A∪B|。当 A 与 B 全空时，约定返回 0.0。
    """
    # TODO: 使用集合的交并集定义，不要直接使用向量点积当相似度。
    a = a.bool()
    b = b.bool()
    intersection = (a&b).sum().item()
    union = (a|b).sum().item()
    if union == 0:
        return 0.0
    return intersection/union
    raise NotImplementedError


def summarize_interactions(interactions: torch.Tensor) -> dict:
    """统计隐式交互矩阵的基本信息。

    返回字典，必须包含以下键：
        n_users: int
        n_items: int
        n_observed: int                 # 被观察到的用户—物品单元数
        sparsity: float                # 1 - n_observed/(n_users*n_items)
        user_counts: list[int]         # 每位用户交互过多少件物品
        item_counts: list[int]         # 每件物品被多少位用户交互过
    假设 n_users、n_items 均大于 0。
    """
    # TODO
    return {
        "n_users":interactions.shape[0],
        "n_items":interactions.shape[1],
        "n_observed":interactions.sum().item(),
        "sparsity":1-interactions.sum().item()/(interactions.shape[0]*interactions.shape[1]),
        "user_counts":interactions.sum(dim=1).tolist(),
        "item_counts":interactions.sum(dim=0).tolist()
    }
    raise NotImplementedError


if __name__ == "__main__":
    events = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]
    print("完成 TODO 后，请自行输出交互矩阵和统计结果：")
    print("events =", events)
