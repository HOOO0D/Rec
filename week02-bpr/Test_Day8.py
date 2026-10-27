"""Day 8 Top-K 评价自动验收。

用法：
    python Test_Day8.py
    python Test_Day8.py --run-dir runs/day7_sgd_ml100k

说明：
    A–E 为不依赖真实数据的五项单元测试，总计 80 分。
    F 必须提供 Day 7 已训练的 MovieLens-100K 实验目录，额外 20 分。
    只有 6/6 全部通过才算 Day 8 自动验收完成。
    理论问答和手算另行验收。
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import traceback
from pathlib import Path

import torch

from evaluate import (
    recall_at_k, hit_at_k, ndcg_at_k, validate_splits,
    mask_seen_scores, topk_items, evaluate_full_ranking,
)


def check(condition: bool, message: str) -> None:
    """断言测试条件，不满足时提供中文说明。"""
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, name: str) -> None:
    """浮点结果采用误差容限，避免把微小舍入差异误判为错误。"""
    check(math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6),
          f"{name}：实际 {actual}，期望 {expected}")


def must_raise_value_error(fn, name: str) -> None:
    """检查非法输入是否明确拒绝，而不是静默吞掉错误。"""
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"{name}：应抛出 ValueError")


def test_a_hand_metrics() -> None:
    """A：指标数学定义和单个用户手算结果。"""
    ranking = [3, 1, 0, 2]
    relevant = {1, 2}
    close(recall_at_k(ranking, relevant, 3), 0.5, "Recall@3")
    close(hit_at_k(ranking, relevant, 3), 1.0, "Hit@3")
    idcg = 1 + 1 / math.log2(3)
    close(ndcg_at_k(ranking, relevant, 3), (1 / math.log2(3)) / idcg, "NDCG@3")
    close(recall_at_k(ranking, relevant, 4), 1.0, "Recall@4")
    close(ndcg_at_k(ranking, relevant, 4),
          (1 / math.log2(3) + 1 / math.log2(5)) / idcg, "NDCG@4")
    close(hit_at_k(ranking, {8}, 4), 0.0, "未命中时的 Hit@K")
    close(ndcg_at_k([9, 8, 7], {7}, 3), 0.5, "第 3 名命中时的 NDCG")
    for fn in (recall_at_k, hit_at_k, ndcg_at_k):
        close(fn(ranking, set(), 3), 0.0, "空正例集合的约定结果")
        must_raise_value_error(lambda f=fn: f(ranking, relevant, 0), "K=0")
    close(recall_at_k([3, 1], relevant, 99), 0.5, "K 大于推荐长度")


def test_b_masks() -> None:
    """B：过滤只能作用于评分副本，必须保留正确物品索引。"""
    scores = torch.tensor([10.0, 5.0, 5.0, 3.0, 2.0])
    before = scores.clone()
    result = mask_seen_scores(scores, {0, 2})
    check(isinstance(result, torch.Tensor), "屏蔽后必须返回 Tensor")
    check(result.shape == scores.shape, "屏蔽不能改变评分长度")
    check(torch.equal(scores, before), "不能修改传入的原始 scores")
    check(torch.isneginf(result[0]) and torch.isneginf(result[2]), "已交互物品必须为 -inf")
    for idx in (1, 3, 4):
        close(result[idx].item(), scores[idx].item(), "未交互物品的分数保持不变")
    empty = mask_seen_scores(scores, set())
    check(torch.equal(empty, scores), "空过滤集合不应改变分数")
    all_masked = mask_seen_scores(scores, set(range(5)))
    check(torch.isneginf(all_masked).all().item(), "全部交互后必须全部被屏蔽")


def test_c_topk() -> None:
    """C：排序稳定性、有限候选与小 K 边界。"""
    scores = torch.tensor([float('-inf'), 5.0, float('-inf'), 3.0, 2.0])
    check(topk_items(scores, 3) == [1, 3, 4], "屏蔽物品不能重新进入 Top-K")
    check(topk_items(scores, 2) == [1, 3], "Top-2 排序错误")
    check(topk_items(scores, 100) == [1, 3, 4], "候选数不足 K 时不能填入无效物品")
    check(topk_items(torch.tensor([0.0, 5.0, 5.0, 5.0]), 3) == [1, 2, 3],
          "相同分数应该按物品 ID 升序稳定排序")
    check(topk_items(torch.full((4,), float('-inf')), 3) == [],
          "全部候选被屏蔽时应返回空列表")
    check(topk_items(torch.tensor([-2.0, -1.0]), 2) == [1, 0],
          "负数评分也是有效评分")
    must_raise_value_error(lambda: topk_items(torch.zeros(3), 0), "非法 K=0")
    must_raise_value_error(lambda: topk_items(torch.zeros(3), -1), "非法 K=-1")


def test_d_split_leakage() -> None:
    """D：同一数据集内重复记录与三份划分交叉都必须拒绝。"""
    train = [(0, 0), (0, 2), (1, 1)]
    valid = [(0, 1), (1, 3)]
    test = [(0, 3), (1, 2)]
    result = validate_splits(train, valid, test)
    check(result == {"n_train": 3, "n_valid": 2, "n_test": 2},
          "数据划分的记录统计不正确")
    must_raise_value_error(lambda: validate_splits(train + [(0, 0)], valid, test),
                           "训练集内部重复")
    must_raise_value_error(lambda: validate_splits(train, valid + [(0, 1)], test),
                           "验证集内部重复")
    must_raise_value_error(lambda: validate_splits(train, valid, test + [(1, 2)]),
                           "测试集内部重复")
    must_raise_value_error(lambda: validate_splits(train, valid + [(0, 0)], test),
                           "训练和验证集交叉")
    must_raise_value_error(lambda: validate_splits(train, valid, test + [(0, 2)]),
                           "训练和测试集交叉")
    must_raise_value_error(lambda: validate_splits(train, valid, test + [(1, 3)]),
                           "验证和测试集交叉")
    must_raise_value_error(lambda: validate_splits([(-1, 2)], valid, test),
                           "负数编号不合法")


class 评分表模型(torch.nn.Module):
    """固定用户—物品评分的模拟模型，用于检验评价器而不依赖模型训练。"""
    def __init__(self, values: list[list[float]]):
        super().__init__()
        self.register_buffer("table", torch.tensor(values, dtype=torch.float32))
        self.called_in_eval = True
        self.called_without_grad = True

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        self.called_in_eval = self.called_in_eval and not self.training
        self.called_without_grad = self.called_without_grad and not torch.is_grad_enabled()
        return self.table[user_ids, item_ids]


def test_e_full_ranking() -> None:
    """E：检查全物品候选、用户宏平均、屏蔽高分训练物品与评估模式。"""
    model = 评分表模型([
        [99, 2, 4, 8, 6],
        [5, 99, 6, 4, 8],
        [1, 1, 1, 1, 1],
    ])
    train = [(0, 0), (1, 1)]
    valid = [(0, 3), (0, 4), (1, 4)]
    output = evaluate_full_ranking(model, train, valid, n_items=5, k=1, item_batch_size=2)
    check(isinstance(output, dict), "全排序评价结果必须为 dict")
    check(output.get("K") == 1 and output.get("n_eval_users") == 2,
          "只统计有验证正例的两位用户，不应把第三位用户计入分母")
    close(output["Recall@K"], 0.75, "两个用户的宏平均 Recall@1")
    close(output["Hit@K"], 1.0, "两个用户的宏平均 Hit@1")
    close(output["NDCG@K"], 1.0, "两个用户的宏平均 NDCG@1")
    check(output.get("candidate_mode") == "full_sort", "必须标记为全物品排序评价")
    check(model.called_in_eval and model.called_without_grad,
          "评价必须使用 model.eval() 和 torch.no_grad()")
    output2 = evaluate_full_ranking(model, train, valid, n_items=5, k=2, item_batch_size=3)
    close(output2["Recall@K"], 1.0, "Top-2 覆盖全部验证正例")
    close(output2["NDCG@K"], 1.0, "Top-2 理想排序")
    must_raise_value_error(
        lambda: evaluate_full_ranking(model, [(0, 3)], valid, n_items=5, k=1),
        "历史交互与留出正例不能交叉")
    must_raise_value_error(
        lambda: evaluate_full_ranking(model, train, [], n_items=5, k=1),
        "没有验证正例时不能伪造分数")


def test_f_day7_integration(
    run_dir: str,
    method: str = "BPR-SGD",
    checkpoint: str | None = None,
) -> None:
    """F：真实 MovieLens-100K Day 7 实验文件的只读验证与结果落盘。"""
    root = Path(run_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    check(config.get("n_users") == 943 and config.get("n_items") == 1682,
          "请提供 Day 7 的 MovieLens-100K 实验，而不是合成数据运行目录")
    counts = [int(config.get(k, -1)) for k in ("n_train", "n_valid", "n_test")]
    check(all(x > 0 for x in counts) and sum(counts) == 100000,
          "MovieLens-100K 的三份划分应覆盖全部 100000 条观测交互")
    # 使用临时输出目录，避免测试覆盖正式保存的模型和实验日志。
    from run_day8 import run_validation
    with tempfile.TemporaryDirectory() as temp:
        result = run_validation(root, temp, k=10, method=method, checkpoint=checkpoint)
        check(result.get("K") == 10 and result.get("n_eval_users", 0) > 0,
              "验证用户数量或 K 错误")
        check(result.get("stage") == "validation", "本日只允许输出验证指标")
        for name in ("Recall@K", "Hit@K", "NDCG@K"):
            check(math.isfinite(float(result[name])) and 0 <= result[name] <= 1,
                  f"{name} 必须在 [0,1] 范围内")
        output = Path(temp)
        for fname in ("validation_metrics.json", "evaluation_config.json",
                      "validation_recommendations.tsv"):
            check((output / fname).is_file(), f"没有生成 {fname}")
        eval_config = json.loads((output / "evaluation_config.json").read_text(encoding="utf-8"))
        check(eval_config.get("candidate_mode") == "full_sort", "候选集必须为全量物品")
        check(eval_config.get("mask_interactions") == "train_only", "验证阶段仅屏蔽 train")
        check(eval_config.get("checkpoint_path") == result["checkpoint_path"],
              "评价配置必须记录实际加载的权重文件")
        check(eval_config.get("optimizer") == result["model_optimizer"],
              "评价配置和指标中的优化器来源必须一致")
        check(not any("test_metric" in f.name for f in output.iterdir()),
              "Day 8 不应提前写出测试集指标")
        train = set()
        for line in (root / "train_pairs.tsv").read_text(encoding="utf-8").splitlines():
            if line.strip():
                u, i = map(int, line.split('\t'))
                train.add((u, i))
        lines = (output / "validation_recommendations.tsv").read_text(encoding="utf-8").splitlines()
        check(lines and lines[0] == "user_id\titem_ids", "推荐文件表头错误")
        for line in lines[1:]:
            u_str, items_str = line.split('\t')
            u = int(u_str)
            recommendations = [int(x) for x in items_str.split(',') if x]
            check(len(recommendations) <= 10 and len(recommendations) == len(set(recommendations)),
                  "推荐物品数或重复项错误")
            check(all((u, i) not in train for i in recommendations),
                  "推荐结果泄漏了训练已交互物品")
        print(f"    模型权重：{result['checkpoint_path']}，轮次：{result['checkpoint_epoch']}")
        print(f"    MovieLens 验证集：Recall@10={result['Recall@K']:.4f}，"
              f"Hit@10={result['Hit@K']:.4f}，NDCG@10={result['NDCG@K']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 8：Full Ranking 与指标自动验收")
    parser.add_argument("--run-dir", default=None,
                        help="可选：Day 7 已完成的真实 MovieLens-100K 实验产物目录")
    parser.add_argument("--method", choices=("BPR-SGD", "BPR-Adam"), default="BPR-SGD",
                        help="对比实验的方法，默认 BPR-SGD")
    parser.add_argument("--checkpoint", default=None,
                        help="权重文件路径；相对路径基于 --run-dir，优先于默认选择")
    args = parser.parse_args()
    tests = [
        ("A. Recall / Hit / NDCG 手算与边界", 20, test_a_hand_metrics),
        ("B. 训练交互屏蔽与不原地修改", 15, test_b_masks),
        ("C. Top-K 排序、同分与候选不足", 15, test_c_topk),
        ("D. 划分去重与交叉泄漏检查", 15, test_d_split_leakage),
        ("E. 固定模型的全物品排序验证", 15, test_e_full_ranking),
    ]
    passed = score = 0
    failed = []
    for name, points, test in tests:
        try:
            test()
            passed += 1
            score += points
            print(f"[PASS] {name} (+{points})")
        except Exception as ex:
            failed.append(name)
            print(f"[FAIL] {name}：{type(ex).__name__}：{ex}")
            traceback.print_exc(limit=2)
    if args.run_dir is None:
        print("[SKIP] F. MovieLens-100K Checkpoint 评价与落盘 (+20)：未传入 --run-dir")
    else:
        try:
            test_f_day7_integration(args.run_dir, method=args.method,
                                   checkpoint=args.checkpoint)
            passed += 1
            score += 20
            print("[PASS] F. MovieLens-100K Checkpoint 评价与落盘 (+20)")
        except Exception as ex:
            failed.append("F. MovieLens-100K Checkpoint 评价与落盘")
            print(f"[FAIL] F. MovieLens-100K Checkpoint 评价与落盘：{type(ex).__name__}：{ex}")
            traceback.print_exc(limit=2)
    print(f"\nDay 8 自动验收：{passed}/6 项，{score}/100 分")
    if score < 100:
        if failed:
            print("未通过项目：" + "；".join(failed))
            print("请根据上方异常修复对应实现，再重新运行验收。")
        if args.run_dir is None:
            print("F 项尚未执行；完整验收还需通过 --run-dir 提供真实 Day 7 MovieLens 实验目录。")
        raise SystemExit(1)
    print("自动测试全部通过，仍需完成纸笔计算与概念问答。")


if __name__ == "__main__":
    main()
