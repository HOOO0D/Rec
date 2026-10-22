"""第 6 天：BPR 概念与损失自动验收，共 100 分。依赖 Python >= 3.10 和 PyTorch。在 bpr_practice.py 所在目录运行 python Test_Day6.py。不要求负采样器、完整 BPR-MF 训练、RecBole、MovieLens 或评估指标，后续再学习这些内容。"""
from __future__ import annotations

import math
import sys
import traceback
import torch

try:
    from bpr_practice import (
        is_valid_triplet, enumerate_triplets, pairwise_scores,
        preference_probability, bpr_loss, bpr_objective,
    )
except Exception as exc:
    print(f"[ERROR] Import bpr_practice.py failed: {exc}")
    raise SystemExit(1)


def ensure(cond, msg):
    if not cond:
        raise AssertionError(msg)


def close(actual, expected, name, atol=1e-6, rtol=1e-6):
    ensure(isinstance(actual, torch.Tensor), f"{name} must be a torch.Tensor")
    ensure(tuple(actual.shape) == tuple(expected.shape),
           f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    ensure(torch.allclose(actual, expected, atol=atol, rtol=rtol),
           f"{name}: got {actual.detach().cpu().tolist()}, expected {expected.tolist()}")


def test_A():
    """20 分：合法三元组、微型集合穷举与边界情况。"""
    positives = {0: {0, 2}, 1: {1}, 2: {0, 1, 2, 3}, 3: set()}
    good = [(0, 0, 1), (0, 2, 3), (1, 1, 0)]
    bad = [(0, 1, 2), (0, 0, 2), (1, 1, 1), (2, 0, 1),
           (3, 0, 1), (4, 0, 1), (0, -1, 3), (0, 0, 4)]
    for triple in good:
        ensure(is_valid_triplet(*triple, positives, 4) is True,
               f"Legitimate triple {triple} rejected")
    for triple in bad:
        ensure(is_valid_triplet(*triple, positives, 4) is False,
               f"Invalid triple {triple} accepted")
    expected = [
        (0, 0, 1), (0, 0, 3), (0, 2, 1), (0, 2, 3),
        (1, 1, 0), (1, 1, 2), (1, 1, 3),
    ]
    original = {u: v.copy() for u, v in positives.items()}
    got = enumerate_triplets(positives, 4)
    ensure(got == expected, f"Expected sorted triples {expected}, got {got}")
    ensure(positives == original, "Do not modify the input positive sets")
    ensure(enumerate_triplets({0: set(), 1: {0, 1}}, 2) == [],
           "Users with no legal positive-negative comparison have no triples")


def test_B():
    """20 分：矩阵分解配对点积与形状。"""
    users = torch.tensor([[1., 2.], [3., -1.]])
    items = torch.tensor([[2., 1.], [1., -2.], [-1., 1.], [0., 3.]])
    triples = torch.tensor([[0, 0, 1], [1, 3, 2]], dtype=torch.long)
    pos, neg, diff = pairwise_scores(users, items, triples)
    close(pos, torch.tensor([4., -3.]), "positive scores")
    close(neg, torch.tensor([-3., -4.]), "negative scores")
    close(diff, torch.tensor([7., 1.]), "score differences")
    single = torch.tensor([[0, 0, 1]], dtype=torch.long)
    p1, n1, d1 = pairwise_scores(users, items, single)
    close(p1, torch.tensor([4.]), "batch=1 positive score")
    close(n1, torch.tensor([-3.]), "batch=1 negative score")
    close(d1, torch.tensor([7.]), "batch=1 score difference")


def test_C():
    """25 分：Sigmoid、log(2)、手算值与均值归约。"""
    delta = torch.tensor([-2., 0., 2.], dtype=torch.float64)
    probs = preference_probability(delta)
    expected_probs = torch.tensor([
        1 / (1 + math.exp(2)), 0.5, 1 / (1 + math.exp(-2))
    ], dtype=torch.float64)
    close(probs, expected_probs, "sigmoid probabilities")
    ensure(probs[0] < probs[1] < probs[2], "Probability should increase with delta")
    loss = bpr_loss(delta)
    ensure(isinstance(loss, torch.Tensor) and loss.ndim == 0,
           "BPR loss must be a scalar tensor")
    expected = sum(math.log1p(math.exp(-float(v))) for v in delta) / delta.numel()
    ensure(abs(loss.item() - expected) < 1e-9,
           f"Mean BPR loss should be {expected}, got {loss.item()}")
    ensure(abs(bpr_loss(torch.tensor([0.], dtype=torch.float64)).item()
               - math.log(2)) < 1e-9, "For delta=0 loss must equal ln(2)")
    ensure(bpr_loss(torch.tensor([2.])).item() <
           bpr_loss(torch.tensor([0.])).item() <
           bpr_loss(torch.tensor([-2.])).item(),
           "BPR loss must decrease as positive-negative score gap increases")


def test_D():
    """20 分：梯度流向用户、正样本物品和负样本物品。"""
    users = torch.tensor([[1.]], requires_grad=True)
    items = torch.tensor([[0.5], [0.]], requires_grad=True)
    triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
    loss = bpr_objective(users, items, triples)
    ensure(isinstance(loss, torch.Tensor) and loss.ndim == 0,
           "bpr_objective should return a scalar tensor")
    reference = math.log1p(math.exp(-0.5))
    ensure(abs(loss.item() - reference) < 1e-6,
           f"Expected {reference}, got {loss.item()}")
    ensure(loss.requires_grad, "BPR objective must preserve autograd")
    loss.backward()
    for name, arr in (("users", users), ("items", items)):
        ensure(arr.grad is not None and bool(torch.isfinite(arr.grad).all()),
               f"{name} needs finite gradients")
    ensure(users.grad[0, 0].item() < 0, "User gradient must be negative here")
    ensure(items.grad[0, 0].item() < 0, "Positive item gradient should be negative")
    ensure(items.grad[1, 0].item() > 0, "Negative item gradient should be positive")
    ensure(torch.allclose(items.grad[0], -items.grad[1], atol=1e-6),
           "Positive and negative item gradients should oppose one another")


def test_E():
    """15 分：极端得分下数值稳定，梯度无 NaN。"""
    delta = torch.tensor([-1000., 0., 1000.], dtype=torch.float64,
                         requires_grad=True)
    p = preference_probability(delta)
    ensure(bool(torch.isfinite(p).all()), "Probabilities should stay finite")
    ensure(bool(((p >= 0) & (p <= 1)).all()), "Probabilities in [0,1]")
    loss = bpr_loss(delta)
    ensure(bool(torch.isfinite(loss)), "BPR loss should stay finite")
    expected = (1000 + math.log(2)) / 3
    ensure(abs(loss.item() - expected) < 1e-8,
           f"Stable extreme-score mean expected about {expected}, got {loss.item()}")
    loss.backward()
    ensure(delta.grad is not None and bool(torch.isfinite(delta.grad).all()),
           "Extreme gradients must be finite")
    ensure(delta.grad[0] < 0 and delta.grad[1] < 0,
           "Loss decreases with positive-negative score difference")


def main():
    torch.manual_seed(20261019)
    cases = [
        ("A. 合法三元组与小样本枚举", 20, test_A),
        ("B. MF 成对点积及形状", 20, test_B),
        ("C. Sigmoid / BPR Loss 手算", 25, test_C),
        ("D. 反向传播与梯度方向", 20, test_D),
        ("E. 极端输入的数值稳定性", 15, test_E),
    ]
    score, passed = 0, 0
    print("=" * 62)
    print("Day 6 | BPR 原理与损失函数自动验收")
    print("=" * 62)
    for name, points, fn in cases:
        try:
            fn()
            score += points
            passed += 1
            print(f"[PASS] {name} (+{points})")
        except Exception as exc:
            print(f"[FAIL] {name} (+0): {type(exc).__name__}: {exc}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
    print("-" * 62)
    print(f"自动验收：{passed}/{len(cases)} 项通过，得分 {score}/100")
    print("理论答辩与论文笔记需另外检查；100 分不等于论文验收已完成。")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
