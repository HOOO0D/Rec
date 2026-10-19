"""Day 4｜推荐系统概念的数据实践自动验收（PyTorch，无额外依赖）。

执行: python Test_Day4.py
说明: 自动测试只覆盖交互数据理解（100 分）；概念笔记与口头问答必须单独验收。
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from interaction_analysis import (
        build_explicit_matrix,
        build_implicit_matrix,
        user_overlap,
        item_overlap,
        jaccard,
        summarize_interactions,
    )
except Exception as exc:
    print("[ERROR] 无法从 interaction_analysis.py 导入约定的 6 个函数。")
    print("详细错误:", repr(exc))
    raise SystemExit(1)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_a_explicit():
    ratings, mask = build_explicit_matrix(2, 3, [(0, 0, 5.0), (0, 1, 0.0), (1, 2, 3.5)])
    check(isinstance(ratings, torch.Tensor) and isinstance(mask, torch.Tensor), "返回值必须是两个 Tensor")
    check(ratings.shape == (2, 3) and mask.shape == (2, 3), "形状必须都是 [2,3]")
    check(ratings.dtype == torch.float32, "显式评分张量应为 float32")
    check(mask.dtype == torch.bool, "观察掩码应为 bool")
    expected_mask = torch.tensor([[True, True, False], [False, False, True]])
    check(torch.equal(mask, expected_mask), "观察掩码错误")
    check(torch.isnan(ratings[~mask]).all().item(), "未观察位置必须是 NaN")
    check(torch.allclose(ratings[mask], torch.tensor([5.0, 0.0, 3.5])), "真实评分不正确，0 分也必须保留")
    empty_ratings, empty_mask = build_explicit_matrix(1, 2, [])
    check(torch.isnan(empty_ratings).all().item() and not empty_mask.any().item(), "空记录边界错误")


def toy():
    return build_implicit_matrix(3, 4, [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (0, 0)])


def test_b_implicit():
    observed = toy()
    expected = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.int64)
    check(isinstance(observed, torch.Tensor), "需要返回 Tensor")
    check(observed.shape == (3, 4), "形状必须为 [3,4]")
    check(torch.equal(observed, expected), "隐式交互矩阵错误；重复事件只能记为 1")
    check(int(observed.sum()) == 6, "观察到 6 个唯一的用户—物品交互")
    empty = build_implicit_matrix(2, 3, [])
    check(empty.shape == (2, 3) and torch.count_nonzero(empty).item() == 0, "空事件边界错误")


def test_c_overlap():
    x = toy()
    check(user_overlap(x, 0, 1) == 1, "u0/u1 应有 1 个共同物品")
    check(user_overlap(x, 0, 2) == 0, "u0/u2 无共同物品")
    check(user_overlap(x, 1, 2) == 1, "u1/u2 应有 1 个共同物品")
    check(item_overlap(x, 1, 2) == 1, "i1/i2 应有 1 个共同用户")
    check(item_overlap(x, 0, 3) == 0, "i0/i3 无共同用户")
    check(item_overlap(x, 1, 1) == 2, "i1 被两个用户交互")
    other = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=torch.int64)
    check(user_overlap(other, 0, 1) == 2 and item_overlap(other, 0, 2) == 2, "不能硬编码第一次矩阵的结果")


def test_d_jaccard():
    x = toy()
    check(math.isclose(jaccard(x[0], x[1]), 1 / 3, abs_tol=1e-6), "用户 0/1 的 Jaccard 应为 1/3")
    check(math.isclose(jaccard(x[:, 1], x[:, 2]), 1 / 3, abs_tol=1e-6), "物品 1/2 的 Jaccard 应为 1/3")
    check(math.isclose(jaccard(x[0], x[2]), 0.0, abs_tol=1e-6), "不重叠时应为 0")
    check(math.isclose(jaccard(x[0], x[0]), 1.0, abs_tol=1e-6), "相同非空向量应为 1")
    check(math.isclose(jaccard(torch.zeros(4), torch.zeros(4)), 0.0, abs_tol=1e-6), "两个空集合时本练习约定返回 0")


def test_e_summary():
    summary = summarize_interactions(toy())
    check(isinstance(summary, dict), "必须返回 dict")
    target = {
        "n_users": 3,
        "n_items": 4,
        "n_observed": 6,
        "user_counts": [2, 2, 2],
        "item_counts": [1, 2, 2, 1],
    }
    for key, expected in target.items():
        check(summary.get(key) == expected, f"统计字段 {key} 不正确，期望 {expected}")
    check(math.isclose(summary.get("sparsity", -1), 0.5, abs_tol=1e-6), "稀疏率应为 0.5")
    other = summarize_interactions(torch.zeros((2, 3), dtype=torch.int64))
    check(other.get("n_observed") == 0, "全未观察矩阵的交互数应为 0")
    check(other.get("user_counts") == [0, 0], "空矩阵 user_counts 错误")
    check(other.get("item_counts") == [0, 0, 0], "空矩阵 item_counts 错误")
    check(math.isclose(other.get("sparsity", -1), 1.0, abs_tol=1e-6), "全未观察矩阵稀疏率应为 1")


TESTS = [
    ("A. 显式评分与未观察掩码", test_a_explicit),
    ("B. 隐式行为与去重", test_b_implicit),
    ("C. 用户/物品共现计数", test_c_overlap),
    ("D. Jaccard 相似度及边界", test_d_jaccard),
    ("E. 数据规模与稀疏率", test_e_summary),
]


def main():
    print("=" * 60)
    print("Day 4 数据实践验收：满分 100；概念笔记另行人工验收")
    print("=" * 60)
    passed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as exc:
            print(f"[FAIL] {name} (0/20): {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[PASS] {name} (+20)")
    print(f"\n自动测试：{passed}/5 项通过，得分 {passed * 20}/100")
    if passed != len(TESTS):
        print("请完成 interaction_analysis.py 中的 TODO 后重试。")
        return 1
    print("数据实践通过；还需完成 notes.md 和不看资料的概念问答。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
