"""Day 14：LightGCN 训练与验证核心函数（中文 TODO 模板）。

与 Day 13 的 LightGCN 类直接衔接；本文件需要独立完成。
Day 14 只在训练集上建图、采样和拟合；验证集只用于模型选择；不使用测试集调参。

教学约定：
  * 正样本为训练交互 (u,i)，负样本在用户训练未交互物品中均匀随机采样。
  * 排序损失为 mean(softplus(neg_score - pos_score))。
  * L2 正则化作用于当前 batch 的三组第 0 层 Embedding，归一化为 1/(2B)。
  * 逐 batch 重新计算 LightGCN 的图传播；不可 detach() 最终 Embedding。
  * 验证采用 Full Ranking，排除 train 已见物品；指标按有效用户做宏平均。
"""
from __future__ import annotations

import random
import torch
from lightgcn_model import LightGCN
import math

def build_seen_sets(train_pairs: list[tuple[int, int]], n_users: int, n_items: int) -> list[set[int]]:
    """返回每个用户训练已见物品的集合；严禁混入验证/测试交互。"""
    # TODO A1：检查训练交互的整数编号是否在范围内。
    if n_users<= 0 or n_items<=0:
        raise ValueError()
    for u,i in train_pairs:
        if u<0 or u>= n_users:
            raise ValueError()
        if i<0 or i>= n_items:
            raise ValueError()
    
    # TODO A1：按用户整理正样本集合；遇到重复 (u,i) 应抛 ValueError。
    pairs = [set() for _ in range(n_users)]
    for u,i in train_pairs:
        if type(u) is not int or type(i) is not int:
            raise ValueError()
        if i in pairs[u]:
            raise ValueError()
        pairs[u].add(i)
    return pairs
    raise NotImplementedError('TODO A1：构建训练正交互集合')


def sample_negative_items(
    user_ids: torch.Tensor,
    seen_sets: list[set[int]],
    n_items: int,
    rng: random.Random,
) -> torch.Tensor:
    """为每个用户采一个训练未见物品，返回 CPU 上的 long 张量 [B]。

    负物品对该用户的合法候选等概率；若某用户已与所有物品交互，必须抛 ValueError。
    采样范围仅排除训练已观察到的正交互（不能借助验证、测试标签）。
    """
    # TODO A2：对每个 user 做均匀随机采样与拒绝采样。
    if type(n_items) is not int or n_items <= 0:
        raise ValueError("物品数量必须是正整数")

    if not isinstance(user_ids, torch.Tensor):
        raise ValueError("user_ids 必须是 Tensor")

    if user_ids.ndim != 1 or user_ids.dtype != torch.long:
        raise ValueError("user_ids 必须是一维 long 张量")
    negatives = []
    for u in user_ids.tolist():
        if not 0 <= u < len(seen_sets):
            raise ValueError("用户 ID 越界")
        seen = seen_sets[u]
        if len(seen)>=n_items:
            raise ValueError()
        j = rng.randrange(n_items)
        while j in seen:
            j = rng.randrange(n_items)
        negatives.append(j)

    return torch.tensor(negatives,dtype=torch.long,device="cpu")
    # TODO A2：确保未采到训练已见物品；处理零候选、非法用户 ID 等情况。
    raise NotImplementedError('TODO A2：负采样')


def calculate_bpr_loss(
    model: LightGCN,
    user_ids: torch.Tensor,
    pos_item_ids: torch.Tensor,
    neg_item_ids: torch.Tensor,
    reg_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (total_loss, ranking_loss, regularization_loss, score_margin)。

    user_ids/pos_item_ids/neg_item_ids：相同长度 [B]，且都在 model 参数所在设备。
    score_margin 是每个样本正负评分之差 [B]，用于监测训练排序准确率。

    为了避免重复传播，仅调用一次 model()，然后在同一组最终 Embedding 上取向量。
    reg_loss 不包括 lambda；total_loss = rank_loss + reg_lambda * reg_loss。
    """
    # TODO B：生成用户/物品最终 Embedding，算两组点积，得 margin。
    
    if not math.isfinite(reg_lambda) or reg_lambda < 0:
        raise ValueError("reg_lambda 必须是有限的非负数")

    B = user_ids.numel()
    if B == 0:
        raise ValueError("Batch 不能为空")
    user_E , item_E = model()
    item_pos = item_E[pos_item_ids]
    item_neg = item_E[neg_item_ids]
    users = user_E[user_ids]
    pos_scores = (users*item_pos).sum(dim=1)
    neg_scores = (users*item_neg).sum(dim=1)
    margin = pos_scores-neg_scores
    # TODO B：用 torch.nn.functional.softplus(-margin).mean() 计算排序损失。
    Loss = torch.nn.functional.softplus(-margin).mean()
    # TODO B：从 model.user_embedding / item_embedding 读取对应第 0 层行。
    user0 = model.user_embedding(user_ids)
    item0Pos = model.item_embedding(pos_item_ids)
    item0Neg = model.item_embedding(neg_item_ids)
    # TODO B：按 1/(2B) 计算三组初始 Embedding 的平方和；保留计算图。
    reg_Loss = (user0.pow(2).sum()+item0Pos.pow(2).sum()+item0Neg.pow(2).sum())/(2*B)
    # TODO B：验证 reg_lambda 为有限、非负值，并检查 Batch 非空。
    total_loss = Loss + reg_lambda*reg_Loss
    return total_loss,Loss,reg_Loss,margin
    raise NotImplementedError('TODO B：BPR 与 L2 损失')


def train_one_epoch(
    model: LightGCN,
    train_pairs: list[tuple[int, int]],
    seen_sets: list[set[int]],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    reg_lambda: float,
    rng: random.Random,
) -> dict:
    """训练一轮，返回 rank_loss/reg_loss/total_loss/pair_accuracy 和 n_train。

    所有 Loss 指标按训练三元组数量加权平均，不采用不等大小 Batch 的简单平均。
    每个 batch：负采样 → optimizer.zero_grad → calculate_bpr_loss → backward → step。
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")

    if not math.isfinite(reg_lambda) or reg_lambda < 0:
        raise ValueError("reg_lambda 必须是有限的非负数")

    if len(train_pairs) == 0:
        raise ValueError("训练集不能为空")

    if len(seen_sets) != model.user_embedding.num_embeddings:
        raise ValueError("seen_sets 长度必须等于用户数量")
    # TODO C：检查参数，随机打乱训练交互的遍历顺序。
    model.train()
    device = model.user_embedding.weight.device
    n_item = model.item_embedding.num_embeddings

    shuffled_pairs = list(train_pairs)
    rng.shuffle(shuffled_pairs)
    sum_rank = 0.0
    sum_reg = 0.0
    sum_total = 0.0
    n_correct = 0
    n_train = 0 
    # TODO C：把正负 ID 移到模型当前设备，不要把邻接矩阵变成 Dense。
    # TODO C：每个 batch 都重新计算前向；不能复用旧的已传播 Embedding。
    for start in range(0,len(shuffled_pairs),batch_size):
        batch = shuffled_pairs[start:start+batch_size]
        pairs = torch.tensor(batch,dtype=torch.long,device="cpu")
        user_ids = pairs[:,0]
        pos_ids = pairs[:,1]
        neg_ids = sample_negative_items(user_ids,seen_sets,n_item,rng)
        user_ids = user_ids.to(device)
        pos_ids = pos_ids.to(device)
        neg_ids = neg_ids.to(device)
        optimizer.zero_grad()
        total ,rank , reg, margin = calculate_bpr_loss(model,user_ids,pos_ids,neg_ids,reg_lambda)
        total.backward()
        optimizer.step()
        B = user_ids.numel()
        sum_rank += rank.item()*B
        sum_reg += reg.item()*B
        sum_total += total.item()*B
        n_correct += (margin.detach() > 0).sum().item()
        n_train += B
    # TODO C：返回加权平均的 Loss、正负排序准确率和样本数。
    return {
    "ranking_loss": sum_rank / n_train,
    "regularization_loss": sum_reg / n_train,
    "total_loss": sum_total / n_train,
    "pair_accuracy": n_correct / n_train,
    "n_train": n_train,
    }
    raise NotImplementedError('TODO C：单轮训练')


def evaluate_full_ranking(
    model: LightGCN,
    train_pairs: list[tuple[int, int]],
    valid_pairs: list[tuple[int, int]],
    *,
    k: int,
    eval_user_batch_size: int = 128,
) -> dict:
    """验证集全物品 Top-K 排序评价。

    每位有效用户的真实相关集是 valid_pairs 中的所有留出正例。
    候选集是该用户所有物品减去训练已见物品；不能加入测试正例作标签。
    返回 recall@k、hit@k、ndcg@k、n_eval_users、k 等统计。
    """

    # TODO D：在无梯度的 eval 模式下计算最终用户/物品 Embedding（只需一次）。
    
    # TODO D：对每批用户做矩阵乘法，训练已见物品评分置为 -inf。
    if type(k) is not int or k <= 0:
        raise ValueError("k 必须是正整数")

    if type(eval_user_batch_size) is not int or eval_user_batch_size <= 0:
        raise ValueError("eval_user_batch_size 必须是正整数")
    n_users = model.user_embedding.num_embeddings
    n_items = model.item_embedding.num_embeddings
    train_seen = build_seen_sets(train_pairs, n_users, n_items)
    valid_seen = [set() for _ in range(n_users)]
    for u, i in valid_pairs:
        if type(u) is not int or type(i) is not int:
            raise ValueError("验证交互 ID 必须是整数")

        if not (0 <= u < n_users and 0 <= i < n_items):
            raise ValueError("验证交互 ID 越界")

        if i in train_seen[u]:
            raise ValueError("训练集与验证集存在重叠交互")

        if i in valid_seen[u]:
            raise ValueError("验证集存在重复交互")

        valid_seen[u].add(i)
    eval_users = [
        u for u in range(n_users)
        if valid_seen[u]
    ]
    if not eval_users:
        raise ValueError("没有可用于评价的验证用户")
    sum_recall = 0.0
    sum_hit = 0.0
    sum_ndcg = 0.0
    model.eval()
    with torch.no_grad():
        user_E,item_E = model()
        device = user_E.device
        for start in range(0, len(eval_users), eval_user_batch_size):
            batch_users = eval_users[start:start + eval_user_batch_size]
            user_ids = torch.tensor(
                batch_users, dtype=torch.long, device=device,
            )

            # [B,d] @ [d,I] → [B,I]
            scores = user_E[user_ids] @ item_E.T

            if not torch.isfinite(scores).all():
                raise ValueError("模型评分包含 NaN 或无穷大")

            # 4. 屏蔽每个用户训练中已见的物品
            for row, u in enumerate(batch_users):
                if train_seen[u]:
                    scores[row, list(train_seen[u])] = -torch.inf

            # 分数降序排列；同分时保留物品 ID 从小到大的顺序
            top_items = torch.argsort(
                scores, dim=1, descending=True, stable=True,
            )[:, :min(k, n_items)].cpu().tolist()

            # 5. 逐用户计算指标
            for row, u in enumerate(batch_users):
                # 候选不足 K 时，不把被屏蔽的物品算作推荐
                recommended = [
                    i for i in top_items[row]
                    if i not in train_seen[u]
                ]

                truth = valid_seen[u]
                hits = [int(i in truth) for i in recommended]
                n_hits = sum(hits)

                recall = n_hits / len(truth)
                hit = float(n_hits > 0)

                dcg = sum(
                    relevant / math.log2(rank + 1)
                    for rank, relevant in enumerate(hits, start=1)
                )

                idcg = sum(
                    1.0 / math.log2(rank + 1)
                    for rank in range(1, min(k, len(truth)) + 1)
                )

                ndcg = dcg / idcg

                sum_recall += recall
                sum_hit += hit
                sum_ndcg += ndcg

    # 6. 对有效用户做宏平均
    n_eval_users = len(eval_users)

    return {
        f"recall@{k}": sum_recall / n_eval_users,
        f"hit@{k}": sum_hit / n_eval_users,
        f"ndcg@{k}": sum_ndcg / n_eval_users,
        "n_eval_users": n_eval_users,
        "k": k,
    }
    # TODO D：对每位有验证正例的用户计算 Recall、Hit、NDCG，然后宏平均。
    # TODO D：验证、训练交互应不重叠；跳过没有验证正例的用户。
    # TODO D：不要使用测试集，也不要让模型在验证时更新参数。
    raise NotImplementedError('TODO D：Full Ranking 评价')
