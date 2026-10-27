"""Day 9：RecBole BPR 代码与实验协议验收（满分 100 分）。

不带参数：静态配置、协议审计函数、对照规则以及已安装的 RecBole 源码检查。
带 --run-dir：增加真实实验的运行结果与 Checkpoint 校验。

未安装 RecBole 时，源码项为 SKIP；未提供真实运行目录时，实验项为 SKIP。
通过静态单测不代表真实 MovieLens-100K 实验已经完成。
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import traceback
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit("缺少 PyYAML，请在 RecBole 环境中安装：python -m pip install pyyaml")

from protocol_audit import (
    AUDIT_FIELDS,
    summarize_day7,
    summarize_recbole,
    compare_protocols,
)

ROOT = Path(__file__).resolve().parent


class SkipCheck(Exception):
    """资源未准备好时跳过验收。"""


def ensure(ok: bool, explanation: str) -> None:
    if not ok:
        raise AssertionError(explanation)


def test_a_config():
    """A：检查基线实验的固定配置，不接受含糊的默认值。"""
    with (ROOT / "recbole_bpr.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ensure(isinstance(cfg, dict), "YAML 必须解析为字典")
    ensure(cfg.get("seed") == 20261020 and cfg.get("reproducibility") is True,
           "随机种子、可复现配置不正确")
    ensure(cfg.get("embedding_size") == 16, "应与 Day 7 对齐 Embedding 维度")
    ensure(str(cfg.get("learner")).lower() == "adam", "初次 RecBole 基线使用 Adam")
    ensure(float(cfg.get("weight_decay", -1)) == 0., "必须明确记录无 optimizer weight decay")
    ensure(cfg.get("epochs", 0) >= 1 and cfg.get("train_batch_size", 0) > 0,
           "Epoch 与 Batch Size 需要为正整数")
    ensure(cfg.get("load_col", {}).get("inter") == ["user_id", "item_id"],
           "本次按已观察评分构造隐式交互，只加载用户和物品 ID")
    neg = cfg.get("train_neg_sample_args", {})
    ensure(neg.get("distribution") == "uniform" and neg.get("sample_num") == 1,
           "RecBole 应配置每条训练正交互采 1 个均匀负物品")
    ev = cfg.get("eval_args", {})
    ensure(ev.get("split", {}).get("RS") == [0.8, 0.1, 0.1],
           "应配置 RecBole 用户分组 8:1:1 随机划分")
    ensure(ev.get("group_by") == "user" and ev.get("order") == "RO" and ev.get("mode") == "full",
           "评价必须使用用户分组、随机顺序和全排序")
    ensure(set(cfg.get("metrics", [])) == {"Recall", "Hit", "NDCG"}, "指标集合不正确")
    ensure(cfg.get("topk") == [10] and cfg.get("valid_metric") == "NDCG@10",
           "必须按验证集 NDCG@10 选模型")


def test_b_normalize():
    """B：检验协议读取不依赖硬编码。"""
    with (ROOT / "recbole_bpr.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    day7_raw = {"optimizer": "SGD", "embedding_dim": 16, "seed": 20261020}
    day7_copy = dict(day7_raw)
    ours = summarize_day7(day7_raw)
    rec = summarize_recbole(cfg)
    ensure(set(AUDIT_FIELDS).issubset(ours) and set(AUDIT_FIELDS).issubset(rec),
           "必须覆盖八个协议字段")
    ensure(ours["optimizer"] == "sgd" and rec["optimizer"] == "adam", "优化器读取错误")
    ensure(ours["embedding_size"] == rec["embedding_size"] == 16, "Embedding 维度读取错误")
    ensure(ours["seed"] == rec["seed"] == 20261020, "种子读取错误")
    ensure(ours["negative_sampling"] == "uniform_ds_bootstrap", "Day 7 采样协议错误")
    ensure(rec["negative_sampling"] == "uniform_negative_per_observed_positive", "RecBole 采样协议错误")
    ensure(ours["split_strategy"] == "frozen_day7_user_split", "Day 7 数据划分协议错误")
    ensure(rec["split_strategy"] == "recbole_user_random_ratio", "RecBole 数据划分协议错误")
    ensure(ours["candidate_mode"] == rec["candidate_mode"] == "full", "Full Ranking 协议读取错误")
    ensure(ours["regularization"] == "sampled_embedding_l2", "Day 7 正则约定错误")
    ensure(rec["regularization"] == "none", "RecBole BPR 基线没有配置显式 L2 正则")
    ensure(ours["id_mapping"] == "zero_based_no_padding", "Day 7 ID 映射错误")
    ensure(rec["id_mapping"] == "recbole_token_mapping_with_padding", "RecBole ID 映射错误")
    ensure(day7_raw == day7_copy, "不得修改输入配置")
    # 修改几项真实字段，检查不是只返回固定字符串。
    other = dict(cfg)
    other["learner"] = "sgd"
    other["seed"] = 42
    other["embedding_size"] = 32
    other["weight_decay"] = 0.1
    other["train_neg_sample_args"] = {"distribution": "popularity", "sample_num": 2}
    other["eval_args"] = {"split": {"LS": "valid_and_test"}, "group_by": "user", "order": "TO", "mode": "uni99"}
    alt = summarize_recbole(other)
    ensure(alt["optimizer"] == "sgd" and alt["seed"] == 42 and alt["embedding_size"] == 32,
           "配置变化时输出没有正确更新")
    ensure(alt["negative_sampling"] == "other" and alt["split_strategy"] == "other",
           "必须从配置判断负采样与划分策略")
    ensure(alt["candidate_mode"] == "uni99" and alt["regularization"] == "optimizer_weight_decay",
           "应能识别采样评价模式和非零 weight decay")


def test_c_compare():
    """C：比较器只输出不一致字段，保留用户可追溯信息。"""
    a = {name: name for name in AUDIT_FIELDS}
    b = dict(a)
    ensure(compare_protocols(a, b) == [], "完全相同时差异应为空")
    b["optimizer"] = "adam"
    b["negative_sampling"] = "item_uniform"
    b["id_mapping"] = "padding"
    original_a = dict(a)
    original_b = dict(b)
    differences = compare_protocols(a, b)
    ensure(isinstance(differences, list), "返回值必须是列表")
    ensure([d.get("field") for d in differences] == ["optimizer", "negative_sampling", "id_mapping"],
           "应按 AUDIT_FIELDS 顺序输出不一致字段")
    for d in differences:
        ensure(set(d) == {"field", "day7", "recbole"}, "差异记录字段错误")
        field = d["field"]
        ensure(d["day7"] == a[field] and d["recbole"] == b[field], "差异值必须准确")
    ensure(a == original_a and b == original_b, "不得修改传入字典")


def test_d_source():
    """D：要求真正安装 RecBole 并能读取源码；不提供假实现替代。"""
    if importlib.util.find_spec("recbole") is None:
        raise SkipCheck("RecBole 未安装，需要新环境运行后再检查")
    try:
        from recbole.model.general_recommender.bpr import BPR
        from recbole.model.loss import BPRLoss
    except Exception as exc:
        raise AssertionError(f"RecBole 导入失败，请检查版本兼容：{exc}") from exc
    src_loss = inspect.getsource(BPR.calculate_loss)
    src_predict = inspect.getsource(BPR.full_sort_predict)
    src_bpr = inspect.getsource(BPRLoss.forward)
    ensure("NEG_ITEM_ID" in src_loss and "self.loss" in src_loss,
           "必须核对内置 BPR 的负样本与 Loss 流程")
    ensure("item_embedding.weight" in src_predict or "item_embedding" in src_predict,
           "Full Ranking 应使用物品 Embedding 表")
    ensure("sigmoid" in src_bpr and "log" in src_bpr, "需要定位 BPRLoss 的数值实现")
    # 源码通过测试不代表已经理解；还需写完整笔记。


def test_e_artifacts(run_dir: Path):
    """E：检验确实由 RecBole API 保存的完整实验结果。"""
    if not run_dir.exists():
        raise SkipCheck(f"未找到真实运行目录：{run_dir}")
    required = ["run_result.json", "resolved_config.json", "environment.json",
                "source_locations.json", "recbole_bpr.yaml", "protocol_comparison.json"]
    for name in required:
        ensure((run_dir / name).is_file(), f"缺少实验产物：{name}")
    checkpoints = list((run_dir / "saved").glob("*.pth"))
    ensure(checkpoints, "缺少 RecBole 保存的 .pth 模型文件")
    results = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    ensure("best_valid_result" in results and "test_result" in results,
           "必须保留框架原始验证/测试报告；不能只保存自行计算的指标")
    for phase in ("best_valid_result", "test_result"):
        values = results[phase]
        ensure(isinstance(values, dict), f"{phase} 应为指标字典")
        for name in ("recall@10", "hit@10", "ndcg@10"):
            ensure(name in values, f"{phase} 缺少 {name}")
            value = float(values[name])
            ensure(math.isfinite(value) and 0 <= value <= 1, f"{phase} 中 {name} 不合理")
    cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    ensure(cfg.get("model") == "BPR" and cfg.get("dataset") == "ml-100k", "不是要求的标准 BPR + ml-100k")
    ensure(cfg.get("learner", "").lower() == "adam", "首次基线的优化器应为 Adam")
    mode = cfg.get("eval_args", {}).get("mode")
    ensure(
    mode == "full"
    or mode == {"valid": "full", "test": "full"},
    "候选集合不是 Full Ranking",
)
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    ensure(env.get("recbole") and env.get("torch") and env.get("numpy"), "缺少环境版本记录")
    review = json.loads((run_dir / "source_locations.json").read_text(encoding="utf-8"))
    ensure(review.get("bpr_calculate_loss_line") and review.get("bpr_loss_forward_line"),
           "必须能够定位内置 calculate_loss 与 BPRLoss.forward 源码")
    audit = json.loads((run_dir / "protocol_comparison.json").read_text(encoding="utf-8"))
    ensure(audit.get("directly_comparable") is False and len(audit.get("differences", [])) >= 4,
           "协议不同，不应被标记为可直接比较")
    changed = {entry["field"] for entry in audit["differences"]}
    ensure({"optimizer", "negative_sampling", "split_strategy", "regularization", "id_mapping"} <= changed,
           "协议审计应至少识别优化器、采样、切分、正则化和 ID 映射的不同")


def main():
    parser = argparse.ArgumentParser(description="Day 9 RecBole BPR 自动验收")
    parser.add_argument("--run-dir", help="真实 RecBole 实验的输出目录")
    args = parser.parse_args()
    tests = [
        ("A. RecBole 配置与 Full Ranking", 15, test_a_config),
        ("B. Day 7/RecBole 协议规范化", 25, test_b_normalize),
        ("C. 协议差异检测", 20, test_c_compare),
        ("D. RecBole 源码定位", 15, test_d_source),
    ]
    score = passed = 0
    for name, points, task in tests:
        try:
            task()
        except SkipCheck as exc:
            print(f"[SKIP] {name}：{exc}")
        except Exception as exc:
            print(f"[FAIL] {name}：{type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
        else:
            score += points
            passed += 1
            print(f"[PASS] {name} (+{points})")
    if args.run_dir is None:
        print("[SKIP] E. 真实 ml-100k 实验及协议审计：尚未提供 --run-dir")
    else:
        try:
            test_e_artifacts(Path(args.run_dir))
        except SkipCheck as exc:
            print(f"[SKIP] E. 真实实验（25 分）：{exc}")
        except Exception as exc:
            print(f"[FAIL] E. 真实实验（25 分）：{type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
        else:
            score += 25
            passed += 1
            print("[PASS] E. 真实 ml-100k 实验、模型与审计报告 (+25)")
    print(f"\n自动验收：{passed}/5 项通过，{score}/100 分")
    print("最终验收还需人工检查源码理解、实验对照表与 Week 2 复盘。")
    if score != 100:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
