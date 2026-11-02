"""Day 15：公平协议、验证选择、层数消融和最终测试协议自动验收。

命令：
  python Test_Day15.py
  python Test_Day15.py --result-dir experiments/day15_ml100k \
      --run-dir <Day7冻结数据目录> --graph-dir experiments/day11_ml100k

A–E：合成/纯逻辑单测共 80 分；F：真实五组实验记录与最终测试审计 20 分。
测试脚本不提供 TODO 的答案，也不会提前读取测试指标用于超参选择。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from pathlib import Path

from day15_analysis import (
    LAYERS, FAIR_FIELDS, audit_common_protocol, extract_valid_result,
    select_best_layer, make_ablation_rows, audit_final_history,
)


def require(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def fail_as_value_error(fn, message: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(message + "：应抛出 ValueError")


def fake_configs() -> dict[int, dict]:
    template = {
        "embedding_dim": 16, "batch_size": 64, "epochs": 8,
        "learning_rate": 0.001, "reg_lambda": 0.0001,
        "topk": 10, "seed": 42, "device": "cpu", "optimizer": "Adam", "weight_decay": 0.0,
        "negative_sampling": "train_unseen_uniform_one_per_positive",
        "l2_convention": "initial_triplet_rows_sum_sq_div_2B",
        "eval": "valid_full_ranking_macro_users_mask_train",
        "train_pairs_sha256": "fixture-train-hash",
        "day7_run_dir": "some_day7", "day11_graph_dir": "some_day11",
    }
    return {k: {**template, "n_layers": k} for k in LAYERS}


def fake_summary(k: int, cfg: dict) -> dict:
    return {
        "best_epoch": 1 + k,
        "best_valid_result": {"recall@10": 0.12 + 0.02*k,
                              "hit@10": 0.40 + 0.03*k,
                              "ndcg@10": 0.10 + 0.015*k,
                              "n_eval_users": 20, "k": 10},
        "train_pairs_sha256": cfg["train_pairs_sha256"],
        "graph_source": "train_only",
    }


def fixture_rows() -> list[dict]:
    cs = fake_configs()
    return [extract_valid_result(k, cs[k], fake_summary(k, cs[k])) for k in LAYERS]


def test_a_protocol():
    cs = fake_configs()
    got = audit_common_protocol(cs)
    require(isinstance(got, dict), "应返回共享协议字典")
    require(all(got[k] == cs[0][k] for k in FAIR_FIELDS), "共享协议字段不完整或不正确")
    for key, new_value in [("seed", 43), ("eval", "sampled"),
                           ("train_pairs_sha256", "wrong-hash"), ("reg_lambda", 0.1),
                           ("negative_sampling", "popularity")]:
        changed = fake_configs()
        changed[3][key] = new_value
        fail_as_value_error(lambda changed=changed: audit_common_protocol(changed), f"未检测到 {key} 不一致")
    removed = fake_configs()
    del removed[4]
    fail_as_value_error(lambda: audit_common_protocol(removed), "未检测到缺少 K=4")
    invalid = fake_configs()
    invalid[2]["n_layers"] = 3
    fail_as_value_error(lambda: audit_common_protocol(invalid), "未检测到层数映射错误")
    invalid = fake_configs()
    del invalid[2]["embedding_dim"]
    fail_as_value_error(lambda: audit_common_protocol(invalid), "未检测到必填配置缺失")


def test_b_metrics():
    cs = fake_configs()
    record = extract_valid_result(2, cs[2], fake_summary(2, cs[2]))
    for name in ("n_layers", "best_epoch", "recall", "hit", "ndcg", "n_eval_users", "train_pairs_sha256"):
        require(name in record, f"比较记录缺少 {name}")
    require(record["n_layers"] == 2 and record["best_epoch"] == 3, "层数或最佳 Epoch 不正确")
    require(math.isclose(record["ndcg"], .13), "NDCG 没有正确提取")
    bad = fake_summary(2, cs[2]); bad["best_epoch"] = 999
    fail_as_value_error(lambda: extract_valid_result(2, cs[2], bad), "最佳 Epoch 越界未拒绝")
    bad = fake_summary(2, cs[2]); bad["train_pairs_sha256"] = "mismatch"
    fail_as_value_error(lambda: extract_valid_result(2, cs[2], bad), "图源哈希不匹配未拒绝")
    bad = fake_summary(2, cs[2]); bad["best_valid_result"]["ndcg@10"] = 1.2
    fail_as_value_error(lambda: extract_valid_result(2, cs[2], bad), "指标越界未拒绝")
    bad = fake_summary(2, cs[2]); bad["best_valid_result"]["n_eval_users"] = 0
    fail_as_value_error(lambda: extract_valid_result(2, cs[2], bad), "有效用户数量为 0 未拒绝")


def test_c_selection():
    rows = fixture_rows()
    require(select_best_layer(rows[::-1]) == 4, "最佳 NDCG 对应 K=4")
    tied = [dict(r) for r in rows]
    tied[1]["ndcg"] = tied[4]["ndcg"]
    require(select_best_layer(tied) == 1, "NDCG 并列时应优先更小 K")
    dirty = [dict(r, **{"test_ndcg": 0.99 if r["n_layers"] == 0 else 0.01}) for r in rows]
    require(select_best_layer(dirty) == 4, "模型选择不能使用测试指标")
    fail_as_value_error(lambda: select_best_layer(rows[:4]), "缺少一个传播层数未拒绝")
    fail_as_value_error(lambda: select_best_layer(rows + [rows[-1]]), "出现重复传播层数未拒绝")


def test_d_gain():
    base = [dict(r) for r in fixture_rows()]
    results = make_ablation_rows(base[::-1])
    require([r["n_layers"] for r in results] == list(LAYERS), "结果表顺序应为 K=0–4")
    require(math.isclose(results[0]["ndcg_gain_pct_vs_k0"], 0.0), "基线相对增益应为 0")
    require(math.isclose(results[2]["ndcg_gain_pct_vs_k0"], 30.0), "K=2 NDCG 应相对提高 30%")
    require(math.isclose(results[2]["recall_gain_pct_vs_k0"], (0.16-0.12)/0.12*100), "Recall 相对增益错误")
    under = [dict(r) for r in base]
    under[3]["ndcg"] = .02
    r = make_ablation_rows(under)
    require(r[3]["ndcg_gain_pct_vs_k0"] < 0, "指标下降必须展示负增益")
    zero = [dict(r) for r in base]
    zero[0]["recall"] = 0.0
    r = make_ablation_rows(zero)
    require(all(x["recall_gain_pct_vs_k0"] is None for x in r), "零基线应避免除 0")
    require(base[0]["recall"] == 0.12, "不得修改原始结果")


def test_e_history():
    train = [(0, 0), (0, 1), (1, 2)]
    valid = [(0, 2), (1, 3)]
    test = [(0, 3), (1, 4)]
    history = audit_final_history(train, valid, test)
    require(set(history) == set(train) | set(valid), "测试屏蔽的历史必须是 Train∪Valid")
    require(not (set(history) & set(test)), "不得将 Test 标签混入屏蔽历史")
    fail_as_value_error(lambda: audit_final_history(train, [(0, 0)], test), "未检测到训练与验证交集")
    fail_as_value_error(lambda: audit_final_history(train, valid, [(1, 3)]), "未检测到验证与测试交集")
    fail_as_value_error(lambda: audit_final_history(train+[(0, 1)], valid, test), "未检测到训练集内部重复")
    require(train == [(0, 0), (0, 1), (1, 2)], "不得修改原始训练输入")


def test_f_integration(result_dir: Path, run_dir: Path, graph_dir: Path):
    """审计真实五组训练+最终结果，不重复训练，不再次评价测试集。"""
    from run_day14 import load_and_audit
    from run_day15 import _gather_runs
    _, _, train, valid, test, _, digest = load_and_audit(run_dir, graph_dir)
    require(len(train) > 0 and len(valid) > 0 and len(test) > 0, "真实三份划分必须非空")
    proto, rows = _gather_runs(result_dir)
    require(proto["train_pairs_sha256"] == digest, "训练数据图源不一致")
    require(proto["day7_run_dir"] == str(run_dir), "训练集目录不一致")
    require(proto["day11_graph_dir"] == str(graph_dir), "图目录不一致")
    selection = json.loads((result_dir / "selection.json").read_text(encoding="utf-8"))
    require(selection["selection_source"] == "validation_only", "模型必须只由验证集选择")
    require(selection["selected_k"] == select_best_layer(rows), "最佳层数与验证集不符")
    require(selection["test_evaluated"] is False, "选择阶段不可评价测试集")
    csv_path = result_dir / "validation_comparison.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        comparison = list(csv.DictReader(f))
    require(len(comparison) == 5, "验证比较表必须包含五组")
    require([int(r["n_layers"]) for r in comparison] == list(LAYERS), "K=0–4 不完整")
    final_path = result_dir / "final_test_results.json"
    require(final_path.is_file(), "请先在验证集选模后运行 --stage final 完成一次最终测试")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    require(final["selected_k"] == selection["selected_k"], "最终测试层数与选模结果不符")
    require(final["candidate_protocol"] == "full_ranking_mask_train_plus_valid", "测试历史候选过滤规则错误")
    require(final["train_pairs_sha256"] == digest, "测试图源摘要不匹配")
    expected = sorted(set((0, final["selected_k"])))
    require(final["evaluated_layers"] == expected, "测试只应包含 MF 基线和验证集最佳模型")
    require(set(final["test_metrics"]) == {f"K{k}" for k in expected}, "最终测试模型集合不符合要求")
    for info in final["test_metrics"].values():
        for metric in ("recall", "ndcg", "hit"):
            key = f"{metric}@{proto['topk']}"
            require(key in info and math.isfinite(float(info[key])), f"最终测试缺少有限的 {key}")


def main():
    p = argparse.ArgumentParser(description="Day 15：层数消融与公平比较验收（100 分）")
    p.add_argument("--result-dir", help="Day 15 统一实验输出目录")
    p.add_argument("--run-dir", help="Day 7 冻结数据目录")
    p.add_argument("--graph-dir", help="Day 11 归一化图目录")
    args = p.parse_args()
    tests = [
        ("A. 五组训练公平协议", 20, test_a_protocol),
        ("B. 验证结果抽取与元数据校验", 15, test_b_metrics),
        ("C. 验证集模型选择", 15, test_c_selection),
        ("D. 0–4 层消融增益计算", 15, test_d_gain),
        ("E. 最终测试历史过滤协议", 15, test_e_history),
    ]
    score = 0
    for title, points, test in tests:
        try:
            test()
            score += points
            print(f"[PASS] {title} (+{points})", flush=True)
        except Exception as exc:
            print(f"[FAIL] {title}: {type(exc).__name__}: {exc}", flush=True)
    if args.result_dir and args.run_dir and args.graph_dir:
        try:
            test_f_integration(Path(args.result_dir), Path(args.run_dir), Path(args.graph_dir))
            score += 20
            print("[PASS] F. 本地冻结数据五组实验与最终测试审计 (+20)")
        except Exception as exc:
            print(f"[FAIL] F. 本地实验审计: {type(exc).__name__}: {exc}")
    else:
        print("[SKIP] F. 本地实验审计：需同时指定 --result-dir、--run-dir、--graph-dir")
    print(f"\nDay 15 验收结果：{score}/100 分")
    expected = 100 if (args.result_dir and args.run_dir and args.graph_dir) else 80
    if score < expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
