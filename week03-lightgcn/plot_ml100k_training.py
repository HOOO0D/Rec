"""独立绘制 ML-100K 的 LightGCN 训练曲线，不修改已有模型或训练代码。

训练并绘图：python -B plot_ml100k_training.py --train --epochs 10
只读已有记录：python -B plot_ml100k_training.py --metrics-csv PATH/epoch_metrics.csv
所有排名指标来自验证集；不使用测试集选模型。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

# 使用无窗口后端，字体缓存写入临时目录。
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rec-learning-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "experiments" / "day14_ml100k_curves"


def read_metrics(path: Path) -> tuple[list[dict[str, float]], int]:
    """读取 Day 14 CSV，拒绝空记录、非法数值和乱序 epoch。"""
    if not path.is_file():
        raise ValueError(f"找不到训练记录：{path}；可用 --train 先训练")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        cutoffs = [name.removeprefix("ndcg@") for name in fields if name.startswith("ndcg@")]
        if len(cutoffs) != 1 or not cutoffs[0].isdigit() or int(cutoffs[0]) <= 0:
            raise ValueError("CSV 必须包含一组 Day 14 的 recall@K、hit@K、ndcg@K")
        k = int(cutoffs[0])
        required = [
            "epoch", "ranking_loss", "regularization_loss", "total_loss",
            "pair_accuracy", f"recall@{k}", f"hit@{k}", f"ndcg@{k}",
        ]
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"CSV 缺少字段：{', '.join(sorted(missing))}")
        rows = []
        previous_epoch = 0
        for line_number, raw in enumerate(reader, start=2):
            try:
                row = {name: float(raw[name]) for name in required}
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV 第 {line_number} 行存在非法数值") from error
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"CSV 第 {line_number} 行包含 NaN 或无穷大")
            epoch = row["epoch"]
            if not epoch.is_integer() or epoch <= previous_epoch:
                raise ValueError("epoch 必须是严格递增的正整数")
            previous_epoch = int(epoch)
            for name in ("pair_accuracy", f"recall@{k}", f"hit@{k}", f"ndcg@{k}"):
                if not 0 <= row[name] <= 1:
                    raise ValueError(f"{name} 必须在 [0,1] 内")
            rows.append(row)
    if not rows:
        raise ValueError("CSV 没有训练记录")
    return rows, k


def plot_metrics(metrics_path: Path, output_dir: Path) -> dict:
    rows, k = read_metrics(metrics_path)
    config_path = metrics_path.parent / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    epochs = [int(row["epoch"]) for row in rows]
    # max 保留第一次出现的最佳值，与 run_day14 的 checkpoint 选择一致。
    best = max(rows, key=lambda row: row[f"ndcg@{k}"])
    best_epoch = int(best["epoch"])
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 180,
    })

    def line(ax, key, label, color, linestyle="-"):
        ax.plot(epochs, [row[key] for row in rows], label=label, color=color,
                linewidth=2, linestyle=linestyle,
                marker="o" if len(rows) <= 30 else None, markersize=3)

    def finish(ax, title, ylabel, bounded=False):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.axvline(best_epoch, color="#64748b", linestyle="--", linewidth=1,
                   label=f"Best validation NDCG: epoch {best_epoch}")
        if bounded:
            ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    def training_panels(axes):
        line(axes[0], "ranking_loss", "BPR ranking loss", "#2563eb")
        line(axes[0], "total_loss", "Total loss", "#f97316", linestyle="--")
        finish(axes[0], "Training loss", "Loss")

        # 保留未乘 lambda 的原始正则项，避免与总损失的量级混淆。
        line(axes[1], "regularization_loss", "Raw batch L2 (before lambda)", "#8b5cf6")
        finish(axes[1], "Initial embedding regularization", "Raw L2 term")

        line(axes[2], "pair_accuracy", "Positive score > negative score", "#059669")
        finish(axes[2], "Training pair accuracy", "Fraction", bounded=True)

    def validation_panels(axes):
        for ax, name, color in zip(axes, ("recall", "hit", "ndcg"),
                                   ("#2563eb", "#059669", "#e11d48")):
            key = f"{name}@{k}"
            line(ax, key, key, color)
            ax.scatter(best_epoch, best[key], color=color, marker="*", s=130,
                       zorder=5, label=f"At selected epoch: {best[key]:.4f}")
            finish(ax, f"Validation {name.upper()}@{k}", "Score")
            upper = max(row[key] for row in rows)
            ax.set_ylim(0, min(1.05, max(upper * 1.15, 0.01)))

    detail = "Validation: full ranking, mask train-seen items; macro average over users"
    settings = ", ".join(
        f"{name}={config[name]}"
        for name in ("embedding_dim", "n_layers", "batch_size", "learning_rate", "reg_lambda")
        if name in config
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    training_panels(axes[0])
    validation_panels(axes[1])
    fig.suptitle("LightGCN on MovieLens-100K\n" + (settings or detail), fontsize=15)
    fig.text(0.5, 0.012, detail, ha="center", fontsize=10, color="#475569")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(plot_dir / "overview.png")
    fig.savefig(plot_dir / "overview.svg")
    plt.close(fig)

    for name, draw in (("training_curves", training_panels),
                       ("validation_curves", validation_panels)):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        draw(axes)
        fig.suptitle("LightGCN on MovieLens-100K", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(plot_dir / f"{name}.png")
        plt.close(fig)

    summary = {
        "dataset": "ml100k", "metrics_csv": str(metrics_path.resolve()),
        "n_recorded_epochs": len(rows), "best_epoch": best_epoch,
        "selection_metric": f"ndcg@{k}",
        "best_validation": {f"{name}@{k}": best[f"{name}@{k}"]
                            for name in ("recall", "hit", "ndcg")},
        "last_epoch_metrics": rows[-1], "experiment_config": config,
        "evaluation": "validation_full_ranking_macro_users_mask_train",
        "test_evaluated": False,
        "figures": ["overview.png", "overview.svg", "training_curves.png",
                    "validation_curves.png"],
    }
    (plot_dir / "plot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(f"Plots saved to: {plot_dir.resolve()}", flush=True)
    print(f"Best validation NDCG@{k}: {best[f'ndcg@{k}']:.6f} (epoch {best_epoch})", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="使用已有 ml100k 划分训练并出图")
    mode.add_argument("--metrics-csv", type=Path, help="只读取已有 Day 14 CSV，不训练")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="独立实验输出目录；图片存入其 plots 子目录")
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT.parent / "week02-bpr" / "runs" / "day7_sgd_ml100k")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "experiments" / "day11_ml100k")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reg-lambda", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.train:
        # 训练目录必须为空，避免覆盖已有实验结果。
        if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
            parser.error("训练输出目录非空，请使用新目录，或用 --metrics-csv 只绘图")
        data_config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
        if str(data_config.get("dataset", "")).lower().replace("-", "") != "ml100k":
            parser.error("--run-dir 必须指向 dataset=ml100k 的冻结数据划分")
        import torch
        from run_day14 import run_day14

        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        print(f"Training ML-100K: epochs={args.epochs}, device={device}", flush=True)
        run_day14(
            args.run_dir, args.graph_dir, args.output_dir,
            embedding_dim=args.embedding_dim, n_layers=args.n_layers,
            batch_size=args.batch_size, epochs=args.epochs,
            learning_rate=args.learning_rate, reg_lambda=args.reg_lambda,
            topk=args.topk, seed=args.seed, device=device,
        )

    metrics_path = args.metrics_csv if args.metrics_csv is not None else args.output_dir / "epoch_metrics.csv"
    plot_metrics(metrics_path, args.output_dir)


if __name__ == "__main__":
    main()
