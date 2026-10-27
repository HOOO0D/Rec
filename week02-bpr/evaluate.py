"""Day 8：推荐系统 Top-K 评价模块（需要独立完成的练习模板）。

评价协议：
1. 使用全物品候选集，不采用随机负样本候选集。
2. 验证阶段只过滤用户在训练集已交互的物品。
3. 排名分数从高到低，同分时物品编号小者优先。
4. 对每个存在验证正例的用户分别计算指标，再对用户做宏平均。
5. 没有验证正例的用户不计入平均值分母。
6. 不使用测试集评价结果指导训练、挑选模型或调整超参数。

本文件所有需要自行补全的部分都标有 TODO，不提供核心算法答案。
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch


def recall_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """单个用户的 Recall@K；相关集合为空时约定返回 0。"""
    if k <= 0:
        raise ValueError("k必须大于0")
    # TODO A：计算前 K 个推荐物品与相关集合的交集，占相关集合的比例。
    sub = []
    if len(relevant_items) == 0:
        return 0.0
    for i in ranked_items[:k]:
        if i in relevant_items:
            sub.append(i)
    # TODO A：k 必须大于 0。注意不要将分母错写成 K。
    return float(len(sub)/len(relevant_items))
    raise NotImplementedError("请独立完成 Recall@K")


def hit_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """单个用户的 Hit@K；命中至少一个相关物品即为 1，否则为 0。"""
    if k <= 0:
        raise ValueError("k必须大于0")
    # TODO A：只判断是否命中，不计算命中比例。
    sub = 0
    for i in ranked_items[:k]:
        if i in relevant_items:
            return 1.0
    return 0.0
    raise NotImplementedError("请独立完成 Hit@K")


def ndcg_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """单个用户的二值相关性 NDCG@K；相关集合为空时返回 0。"""
    if k <= 0:
        raise ValueError("k必须大于0")
    # TODO A：DCG 中第 r 名（从 1 开始）对应增益 1 / log2(r+1)。
    DCG = 0.0
    IDCG = 0.0
    for r,i in enumerate(ranked_items[:k], start=1):
        if i in relevant_items:
            DCG += 1.0/math.log2(r+1)
    # TODO A：IDCG 只包含 min(k, 相关物品数量) 个理想命中位置。
    IDCG = sum(1/math.log2(r+1) for r in range(1,min(k,len(relevant_items))+1))
    # TODO A：计算 DCG / IDCG，正确处理 k 大于推荐列表长度的情况。
    return DCG/IDCG if IDCG>0 else 0.0
    raise NotImplementedError("请独立完成 NDCG@K")


def validate_splits(
    train_pairs: list[tuple[int, int]],
    valid_pairs: list[tuple[int, int]],
    test_pairs: list[tuple[int, int]],
) -> dict[str, int]:
    """检查训练、验证、测试交互在每个集合内部不重复，集合之间两两不重叠。"""
    # TODO B：检查 (u, i) 的格式和非负整数编号。
    def check_pairs(pairs,name):
        seen = set()
        for pair in pairs:
            if not isinstance(pair,(tuple,list)) or len(pair)!=2:
                raise ValueError()
            u,i = pair
            if type(u) is not int or type(i) is not int:
                raise ValueError()
            if u<0 or i<0:
                raise ValueError()
            if (u, i) in seen:
                raise ValueError(f"{name}内部有重复交互：{(u, i)}")
            seen.add((u,i))
        return seen
    # TODO B：任何一份数据内部重复，或三份数据交叉，都应抛 ValueError。
    train_sets = check_pairs(train_pairs, "训练集")
    test_sets = check_pairs(test_pairs, "测试集")
    valid_sets = check_pairs(valid_pairs, "验证集")
    if test_sets & train_sets:
        raise ValueError()
    if train_sets & valid_sets:
        raise ValueError()
    if valid_sets & test_sets:
        raise ValueError()
    
    # TODO B：返回 n_train、n_valid、n_test 三项原始记录数量。
    return {
        "n_train": len(train_pairs),
        "n_valid": len(valid_pairs),
        "n_test": len(test_pairs),
    }
    raise NotImplementedError("请独立完成三份数据的泄漏检查")


def mask_seen_scores(scores: torch.Tensor, seen_items: set[int]) -> torch.Tensor:
    """复制一维评分向量，将训练已交互物品的分数置为负无穷。"""
    # TODO C：返回新张量；禁止原地修改调用方传入的 scores。
    masked_scores = scores.clone()
    for item_id in seen_items:
        masked_scores[item_id]=float("-inf")
    return masked_scores
    # TODO C：不能删除向量元素，否则物品编号与下标无法对应。
    raise NotImplementedError("请独立完成训练已交互物品屏蔽")


def topk_items(scores: torch.Tensor, k: int) -> list[int]:
    """返回前 K 个有效物品下标；跳过 -inf，同分时按物品编号升序排列。"""
    # TODO D：k 必须大于 0。不能将已屏蔽的 -inf 物品补进推荐列表。
    if k <=0 :
        raise ValueError()
    idx = []
    score_list = scores.tolist()
    candidates = [
        item_id
        for item_id, score in enumerate(score_list)
        if score != float("-inf")
    ]

    # 先按分数降序，再按物品编号升序
    candidates.sort(
        key=lambda item_id: (-score_list[item_id], item_id)
    )
    return candidates[:k]
    # TODO D：候选数不足 K 时只返回有效候选；不允许重复物品。
    raise NotImplementedError("请独立完成稳定排序与 Top-K 推荐")


def evaluate_full_ranking(
    model: torch.nn.Module,
    seen_pairs: list[tuple[int, int]],
    ground_truth_pairs: list[tuple[int, int]],
    n_items: int,
    k: int = 10,
    item_batch_size: int = 512,
) -> dict:
    """对留出正例执行全物品 Top-K 评价，并返回用户宏平均指标。

    参数说明：
        model：支持 score(user_ids[B], item_ids[B]) -> scores[B] 的模型。
        seen_pairs：需要从候选中屏蔽的历史交互；验证阶段只能是 train。
        ground_truth_pairs：留出正例；验证阶段只能是 valid。
        n_items：物品总数量，合法物品编号为 [0,n_items)。
        k：推荐列表长度。
        item_batch_size：一次评分的物品数量；防止生成过大的临时张量。

    返回：
        {"Recall@K", "Hit@K", "NDCG@K", "n_eval_users", "K", "candidate_mode"}

    注意：本函数不得调用 optimizer、backward，不得更新模型参数。
    """
    if n_items <= 0 or k <= 0 or item_batch_size <= 0:
        raise ValueError("n_items、k 和 item_batch_size 必须大于 0")

    validate_splits(seen_pairs, ground_truth_pairs, [])
    if not ground_truth_pairs:
        raise ValueError("没有留出正例，无法评价")

    for _, item_id in seen_pairs + ground_truth_pairs:
        if item_id >= n_items:
            raise ValueError("物品编号超过 n_items 范围")

    seen_by_user = defaultdict(set)
    relevant_by_user = defaultdict(set)
    for u, i in seen_pairs:
        seen_by_user[u].add(i)
    for u, i in ground_truth_pairs:
        relevant_by_user[u].add(i)

    # 评分表测试模型只有 buffer，没有 parameter。
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    device = reference.device if reference is not None else torch.device("cpu")

    recall_sum = 0.0
    hit_sum = 0.0
    ndcg_sum = 0.0

    model.eval()
    with torch.no_grad():
        # 只评价有留出正例的用户；分批评分仍覆盖全部物品。
        for u in sorted(relevant_by_user):
            score_parts = []
            for start in range(0, n_items, item_batch_size):
                end = min(start + item_batch_size, n_items)
                item_ids = torch.arange(start, end, device=device)
                user_ids = torch.full_like(item_ids, u)
                batch_scores = model.score(user_ids, item_ids)
                score_parts.append(batch_scores.cpu())

            scores = torch.cat(score_parts)
            masked_scores = mask_seen_scores(scores, seen_by_user[u])
            ranked_items = topk_items(masked_scores, k)

            relevant = relevant_by_user[u]
            recall_sum += recall_at_k(ranked_items, relevant, k)
            hit_sum += hit_at_k(ranked_items, relevant, k)
            ndcg_sum += ndcg_at_k(ranked_items, relevant, k)

    n_eval_users = len(relevant_by_user)
    return {
        "Recall@K": recall_sum / n_eval_users,
        "Hit@K": hit_sum / n_eval_users,
        "NDCG@K": ndcg_sum / n_eval_users,
        "n_eval_users": n_eval_users,
        "K": k,
        "candidate_mode": "full_sort",
    }
