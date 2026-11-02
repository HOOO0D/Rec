"""Day 15：五组公平对照消融 + 独立最终测试。注释及日志均为中文。

用法：
  python run_day15.py --stage sweep --run-dir <Day7> --graph-dir <Day11> --output-dir <Day15>
  python run_day15.py --stage final --run-dir <Day7> --graph-dir <Day11> --output-dir <Day15>

依赖：已完成 Day 13/14 中的 LightGCN、负采样、BPR 与 Full Ranking。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch

from day15_analysis import (
    LAYERS, FAIR_FIELDS, audit_common_protocol, extract_valid_result,
    select_best_layer, make_ablation_rows, audit_final_history,
)
from run_day14 import run_day14, load_and_audit
from lightgcn_model import LightGCN
from lightgcn_train import evaluate_full_ranking


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("比较结果不能为空")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _gather_runs(output_dir: Path) -> tuple[dict[int, dict], list[dict]]:
    configs, summaries, rows = {}, {}, []
    for k in LAYERS:
        d = output_dir / f"K{k}"
        configs[k] = _read_json(d / "experiment_config.json")
        summaries[k] = _read_json(d / "run_summary.json")
    protocol = audit_common_protocol(configs)
    for k in LAYERS:
        rows.append(extract_valid_result(k, configs[k], summaries[k]))
        path = output_dir / f"K{k}" / "best_valid_model.pt"
        if not path.is_file():
            raise FileNotFoundError(f"K{k} 缺少最佳模型权重：{path}")
    return protocol, rows


def _assert_existing_matches(path: Path, args: argparse.Namespace, k: int, digest: str) -> None:
    """存在旧实验时只允许同协议复用，不能悄悄混用此前不同超参的结果。"""
    cfg = _read_json(path / "experiment_config.json")
    expect = {
        "embedding_dim": args.embedding_dim, "n_layers": k,
        "batch_size": args.batch_size, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "reg_lambda": args.reg_lambda,
        "topk": args.topk, "seed": args.seed, "device": args.device,
        "train_pairs_sha256": digest,
        "day7_run_dir": str(Path(args.run_dir)), "day11_graph_dir": str(Path(args.graph_dir)),
    }
    for key, value in expect.items():
        if cfg.get(key) != value:
            raise ValueError(f"现有 K{k} 与指定实验协议不符：{key} 实际={cfg.get(key)} 期望={value}，请换输出目录")
    for name in ("run_summary.json", "best_valid_model.pt", "epoch_metrics.csv"):
        if not (path / name).is_file():
            raise ValueError(f"现有 K{k} 的结果不完整：缺少 {name}；请在新的输出目录重新运行")


def run_sweep(args: argparse.Namespace) -> dict:
    if args.epochs < 1 or args.topk < 1 or args.embedding_dim < 1 or args.batch_size < 1:
        raise ValueError("epochs、topk、embedding_dim 和 batch_size 必须为正整数")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("学习率必须为有限正数")
    if args.reg_lambda < 0 or not math.isfinite(args.reg_lambda):
        raise ValueError("正则化参数必须为有限非负数")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境不支持 CUDA，请使用 --device cpu")
    _, _, _, _, _, _, digest = load_and_audit(Path(args.run_dir), Path(args.graph_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "final_test_results.json").exists():
        raise ValueError("本目录已经包含最终测试结果，不能再修改模型选择和训练；请使用新目录")
    for k in LAYERS:
        d = output_dir / f"K{k}"
        if d.exists():
            _assert_existing_matches(d, args, k, digest)
            print(f"K={k} 已有相同协议的完整训练记录，复用", flush=True)
        else:
            print(f"\n===== 开始 K={k} 的训练和验证 =====", flush=True)
            run_day14(args.run_dir, args.graph_dir, d,
                      embedding_dim=args.embedding_dim, n_layers=k,
                      batch_size=args.batch_size, epochs=args.epochs,
                      learning_rate=args.learning_rate, reg_lambda=args.reg_lambda,
                      topk=args.topk, seed=args.seed, device=args.device)
    protocol, rows = _gather_runs(output_dir)
    layer = select_best_layer(rows)
    ablation = make_ablation_rows(rows)
    _write_csv(output_dir / "validation_comparison.csv", ablation)
    selection = {
        "selection_source": "validation_only",
        "valid_metric": f"ndcg@{args.topk}",
        "selected_k": layer,
        "mf_baseline_k": 0,
        "tie_break": "smaller_n_layers",
        "train_pairs_sha256": digest,
        "test_evaluated": False,
        "shared_protocol": protocol,
    }
    _write_json(output_dir / "selection.json", selection)
    print(f"五组验证完成：选择 K={layer}，测试集尚未用于指标计算。", flush=True)
    return selection


def _restore_best_model(d: Path, n_users: int, n_items: int, adj: torch.Tensor,
                        dim: int, k: int, digest: str, device: str) -> torch.nn.Module:
    path = d / "best_valid_model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("graph_sha256") != digest:
        raise ValueError(f"{path} 的冻结训练图摘要不匹配")
    cfg = checkpoint.get("model_config", {})
    if (cfg.get("n_users"), cfg.get("n_items"), cfg.get("embedding_dim"), cfg.get("n_layers")) != (n_users, n_items, dim, k):
        raise ValueError(f"{path} 模型维度/层数与选定设置不一致")
    model = LightGCN(n_users, n_items, dim, k, adj).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def run_final(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    if (output_dir / "final_test_results.json").exists():
        raise FileExistsError("已经进行过最终测试；为避免反复查看测试集选模型，本目录禁止重复运行")
    selection = _read_json(output_dir / "selection.json")
    if selection.get("selection_source") != "validation_only" or selection.get("test_evaluated") is not False:
        raise ValueError("模型必须仅根据验证集选出，且此前尚未执行最终测试")
    protocol, rows = _gather_runs(output_dir)
    k_best = select_best_layer(rows)
    if selection.get("selected_k") != k_best or protocol != selection.get("shared_protocol"):
        raise ValueError("当前训练结果与先前的验证集选择不一致")
    config, _, train, valid, test, adj, digest = load_and_audit(Path(args.run_dir), Path(args.graph_dir))
    if protocol["train_pairs_sha256"] != digest or selection["train_pairs_sha256"] != digest:
        raise ValueError("冻结训练集合不一致")
    if (protocol["day7_run_dir"] != str(Path(args.run_dir)) or
            protocol["day11_graph_dir"] != str(Path(args.graph_dir))):
        raise ValueError("最终测试的冻结数据目录与训练时不同")
    if not test:
        raise ValueError("最终测试正例不能为空")
    test_history = audit_final_history(train, valid, test)
    dim = int(protocol["embedding_dim"])
    k_metric = int(protocol["topk"])
    n_users, n_items = int(config["n_users"]), int(config["n_items"])
    # 训练图仍旧是 Day 11 train-only 图，不允许为测试重新加入验证边。
    # 仅测试排序候选集屏蔽 Train∪Valid；Test 只提供留出的真实正例标签。
    evaluated_layers = sorted(set((0, k_best)))
    results = {}
    for k in evaluated_layers:
        model = _restore_best_model(output_dir / f"K{k}", n_users, n_items, adj, dim, k, digest, args.device)
        model.eval()
        with torch.inference_mode():
            result = evaluate_full_ranking(model, test_history, test,
                                           k=k_metric, eval_user_batch_size=128)
        if not all(math.isfinite(float(v)) for v in result.values() if isinstance(v, (int, float))):
            raise ValueError("最终测试存在无穷大或 NaN 指标")
        results[f"K{k}"] = result
        print(f"最终测试 K={k}: {result}", flush=True)
    report = {
        "selected_k": k_best,
        "baseline_k": 0,
        "evaluated_layers": evaluated_layers,
        "test_metrics": results,
        "metric_topk": k_metric,
        "candidate_protocol": "full_ranking_mask_train_plus_valid",
        "test_labels_from": "test_pairs.tsv",
        "graph_source": "train_only",
        "train_pairs_sha256": digest,
        "test_user_count": len({u for u, _ in test}),
        "note": "先在验证集选择 K/最佳 Epoch；仅在最终测试阶段评价选中的 K 与 K=0 基线。",
    }
    _write_json(output_dir / "final_test_results.json", report)
    # 不修改此前已冻结的选择文件，避免破坏可追溯性。
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 15：LightGCN 统一协议的层数消融与最终测试")
    parser.add_argument("--stage", choices=["sweep", "final"], required=True)
    parser.add_argument("--run-dir", required=True, help="Day 7 的冻结数据目录")
    parser.add_argument("--graph-dir", required=True, help="Day 11 的归一化训练图目录")
    parser.add_argument("--output-dir", required=True, help="Day 15 实验输出目录")
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reg-lambda", type=float, default=0.0001)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    outcome = run_sweep(args) if args.stage == "sweep" else run_final(args)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
