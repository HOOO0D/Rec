"""Day 8：将已完成的 Day 7 BPR-MF Checkpoint 接入验证集评价。

本文件提供文件读取、模型恢复和实验记录的工程脚手架。
核心评价函数由你在 evaluate.py 中独立实现。

示例：
    python run_day8.py --run-dir runs/day7_sgd_ml100k \\
        --output-dir runs/day8_sgd_valid --k 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate import (
    evaluate_full_ranking,
    mask_seen_scores,
    topk_items,
    validate_splits,
)


def read_pairs(path: Path) -> list[tuple[int, int]]:
    """读取 Day 7 保存的零基用户—物品对。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到交互文件：{path}")
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path} 第 {line_no} 行不是两个制表符分隔的整数")
            try:
                pairs.append((int(fields[0]), int(fields[1])))
            except ValueError as ex:
                raise ValueError(f"{path} 第 {line_no} 行包含非法编号") from ex
    return pairs


def run_validation(
    run_dir: str | Path,
    output_dir: str | Path,
    k: int = 10,
    method: str = "BPR-SGD",
    checkpoint: str | Path | None = None,
) -> dict:
    """读取模型与冻结的三份数据，只在验证集上计算指标。"""
    if method not in ("BPR-SGD", "BPR-Adam"):
        raise ValueError("method 必须为 BPR-SGD 或 BPR-Adam")
    # 读取 Day 7 已保存的实验元数据和数据划分。
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到实验配置：{config_path}")
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_dir / checkpoint_path
    elif (run_dir / "checkpoint.pt").is_file():
        # 兼容原来的 Day 7 单模型目录。
        checkpoint_path = run_dir / "checkpoint.pt"
    else:
        # 对比实验共用根目录的数据划分，按方法选择模型权重。
        checkpoint_path = run_dir / method / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到模型权重：{checkpoint_path}；可用 --checkpoint 指定权重文件"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train = read_pairs(run_dir / "train_pairs.tsv")
    valid = read_pairs(run_dir / "valid_pairs.tsv")
    test = read_pairs(run_dir / "test_pairs.tsv")

    # 读取测试集仅用于确认三份数据没有交集；绝不对测试集计算指标。
    sizes = validate_splits(train, valid, test)
    for name, field in (("n_train", "n_train"), ("n_valid", "n_valid"), ("n_test", "n_test")):
        if sizes[name] != config[field]:
            raise ValueError(f"冻结划分与配置数量不一致：{name}")
    n_users = int(config["n_users"])
    n_items = int(config["n_items"])
    embedding_dim = int(config["embedding_dim"])
    if not valid:
        raise ValueError("验证集为空，无法开展 Day 8 指标验收")
    for u, i in train + valid + test:
        if not (0 <= u < n_users and 0 <= i < n_items):
            raise ValueError(f"交互编号超过模型范围：({u},{i})")

    # 直接复用 BPRMF 模型和已保存的权重，不在本脚本中重新训练。
    from bpr_model import BPRMF
    model = BPRMF(n_users, n_items, embedding_dim)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" not in saved:
        raise ValueError("Checkpoint 中不存在 state_dict")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()

    # 优化器来源以实际选中的方法目录为准。
    optimizer_name = config.get("optimizer", "unknown")
    if checkpoint_path.parent.name in ("BPR-SGD", "BPR-Adam"):
        optimizer_name = checkpoint_path.parent.name.split("-", 1)[1].lower()
    checkpoint_epoch = saved.get("epoch")
    if checkpoint_epoch is None and checkpoint_path.name in ("checkpoint.pt", "final.pt"):
        checkpoint_epoch = config.get("epochs")

    # 评价时屏蔽 train，保留 valid 正例用于命中统计；不计算 test 指标。
    metrics = evaluate_full_ranking(model, train, valid, n_items=n_items, k=k)
    metrics["stage"] = "validation"
    metrics["model_optimizer"] = optimizer_name
    metrics["checkpoint_path"] = str(checkpoint_path.resolve())
    metrics["checkpoint_epoch"] = checkpoint_epoch

    # 同时保存部分验证用户的 Top-K 列表，便于人工排查物品过滤问题。
    seen_by_user = {}
    for u, i in train:
        seen_by_user.setdefault(u, set()).add(i)
    eval_users = sorted({u for u, _ in valid})
    recommendation_lines = ["user_id\titem_ids"]
    device = next(model.parameters()).device
    with torch.no_grad():
        for u in eval_users[:10]:
            scores_parts = []
            for st in range(0, n_items, 512):
                ids = torch.arange(st, min(st + 512, n_items), device=device)
                user_ids = torch.full_like(ids, u)
                scores_parts.append(model.score(user_ids, ids).detach().cpu())
            scores = torch.cat(scores_parts)
            masked = mask_seen_scores(scores, seen_by_user.get(u, set()))
            recommended = topk_items(masked, k)
            recommendation_lines.append(f"{u}\t{','.join(map(str, recommended))}")

    # 固定候选为全量物品；只保存验证指标，不输出测试指标。
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "validation_recommendations.tsv").write_text(
        "\n".join(recommendation_lines) + "\n", encoding="utf-8"
    )
    eval_config = {
        "stage": "validation", "candidate_mode": "full_sort",
        "mask_interactions": "train_only", "k": k,
        "n_users": n_users, "n_items": n_items,
        "day7_run_dir": str(run_dir),
        "seed": config.get("seed"), "optimizer": optimizer_name,
        "checkpoint_path": metrics["checkpoint_path"],
        "checkpoint_epoch": checkpoint_epoch,
        "metric_average": "macro_by_users_with_validation_positives",
        "tie_break": "item_id_ascending",
    }
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(eval_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 8：BPR-MF 全排序验证集评估")
    parser.add_argument("--run-dir", required=True, help="Day 7 的实验产物目录")
    parser.add_argument("--output-dir", required=True, help="Day 8 的输出目录")
    parser.add_argument("--k", type=int, default=10, help="推荐物品数量，默认 10")
    parser.add_argument("--method", choices=("BPR-SGD", "BPR-Adam"), default="BPR-SGD",
                        help="对比实验的方法，默认 BPR-SGD；原 Day 7 目录仍使用 checkpoint.pt")
    parser.add_argument("--checkpoint", default=None,
                        help="权重文件路径；相对路径基于 --run-dir，优先于默认选择")
    args = parser.parse_args()
    results = run_validation(args.run_dir, args.output_dir, args.k,
                             method=args.method, checkpoint=args.checkpoint)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
