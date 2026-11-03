"""Day 15 独立实验工具：层数对比、MF 对照、多学习率搜索。

python week03-lightgcn/day15_experiments.py --stage plot
python week03-lightgcn/day15_experiments.py --stage sweep --lrs 0.0003 0.001 0.003
python week03-lightgcn/day15_experiments.py --stage final

复用 run_day15 的训练与最终评价；plot 仅读取记录，不重新训练或评价。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from day15_analysis import (
    LAYERS, FAIR_FIELDS, audit_common_protocol, extract_valid_result, select_best_layer,
)
from plot_ml100k_training import read_metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
DEFAULT_SWEEP = ROOT / "experiments" / "day15_lr_sweep"
COLORS = ("#475569", "#2563eb", "#059669", "#ea580c", "#9333ea")
TRAIN_KEYS = ("ranking_loss", "total_loss", "regularization_loss", "pair_accuracy")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Group:
    directory: Path
    protocol: dict
    records: list[dict]
    histories: dict[int, list[dict]]
    selected_k: int

    def checkpoint_row(self, k: int) -> dict:
        epoch = self.records[k]["best_epoch"]
        return self.histories[k][epoch - 1]

    def table_row(self, k: int) -> dict:
        epoch_row = self.checkpoint_row(k)
        topk = self.protocol["topk"]
        row = {
            "learning_rate": self.protocol["learning_rate"], "n_layers": k,
            "best_epoch": self.records[k]["best_epoch"],
            **{f"train_{name}": epoch_row[name] for name in TRAIN_KEYS},
            **{f"valid_{name}@{topk}": self.records[k][name]
               for name in ("recall", "hit", "ndcg")},
        }
        for name in ("recall", "hit", "ndcg"):
            base = self.records[0][name]
            row[f"{name}_gain_pct_vs_same_lr_k0"] = (
                (self.records[k][name] - base) / base * 100 if base else None
            )
        return row


def load_group(directory: Path) -> Group:
    """只根据配置、验证结果和逐轮 CSV 审计与选模，不读取测试指标。"""
    configs = {k: read_json(directory / f"K{k}" / "experiment_config.json") for k in LAYERS}
    protocol = audit_common_protocol(configs)
    records, histories = [], {}
    for k in LAYERS:
        run_dir = directory / f"K{k}"
        summary = read_json(run_dir / "run_summary.json")
        record = extract_valid_result(k, configs[k], summary)
        history, topk = read_metrics(run_dir / "epoch_metrics.csv")
        if topk != protocol["topk"] or [int(r["epoch"]) for r in history] != list(
            range(1, protocol["epochs"] + 1)
        ):
            raise ValueError(f"{run_dir} 的指标 cutoff 或训练轮数与配置不一致")
        best = max(history, key=lambda r: r[f"ndcg@{topk}"])
        if int(best["epoch"]) != record["best_epoch"]:
            raise ValueError(f"{run_dir} 的 best_epoch 与 CSV 的验证最优轮次不一致")
        for name in ("recall", "hit", "ndcg"):
            if not math.isclose(best[f"{name}@{topk}"], record[name], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{run_dir} 的 {name} 摘要与 CSV 不一致")
        if not (run_dir / "best_valid_model.pt").is_file():
            raise FileNotFoundError(f"{run_dir} 缺少最佳模型权重")
        records.append(record)
        histories[k] = history
    selected_k = select_best_layer(records)
    selection = read_json(directory / "selection.json")
    if (selection.get("selection_source") != "validation_only"
            or selection.get("test_evaluated") is not False
            or selection.get("selected_k") != selected_k
            or selection.get("shared_protocol") != protocol):
        raise ValueError(f"{directory} 的 selection.json 与验证记录不一致")
    return Group(directory, protocol, records, histories, selected_k)


def save_figure(fig, path: Path, *, svg: bool = False) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    if svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def draw_curves(series: list[tuple[str, list[dict], int]], topk: int,
                path: Path, title: str, note: str) -> None:
    panels = [
        ("ranking_loss", "Training BPR loss"),
        ("total_loss", "Training total loss"),
        ("regularization_loss", "Training raw L2 (before lambda)"),
        ("pair_accuracy", "Training pair accuracy"),
        *[(f"{m}@{topk}", f"Validation {m.upper()}@{topk}")
          for m in ("recall", "hit", "ndcg")],
    ]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    for ax, (key, label) in zip(axes.flat, panels):
        for idx, (name, history, best_epoch) in enumerate(series):
            color = COLORS[idx % len(COLORS)]
            ax.plot([r["epoch"] for r in history], [r[key] for r in history],
                    label=name, color=color, linewidth=1.8)
            ax.scatter(best_epoch, history[best_epoch - 1][key], color=color, s=32, zorder=4)
        ax.set(title=label, xlabel="Epoch", ylabel="Loss" if "loss" in key else "Score")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.9, note, va="top", fontsize=11, linespacing=1.7,
                       transform=axes.flat[-1].transAxes)
    fig.suptitle(title, fontsize=16)
    fig.text(0.5, 0.01, "Dots: each run's best validation NDCG epoch. Loss: training epoch average.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_figure(fig, path, svg=True)


def draw_bars(rows: list[dict], labels: list[str], panels: list[tuple[str, str]],
              path: Path, title: str) -> None:
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    for ax, (key, label) in zip(axes.flat, panels):
        values = [row[key] for row in rows]
        bars = ax.bar(labels, values, color=COLORS[:len(rows)], width=0.55)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set(title=label, ylim=(0, max(max(values) * 1.25, 0.01)))
        ax.grid(axis="y", alpha=0.2)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, path)


def read_final_metrics(group: Group) -> dict | None:
    """仅绘制已保存的测试指标；不计算测试 Loss，也不据此选模型。"""
    path = group.directory / "final_test_results.json"
    if not path.exists():
        return None
    report = read_json(path)
    expected = sorted({0, group.selected_k})
    if (report.get("selected_k") != group.selected_k
            or report.get("evaluated_layers") != expected
            or report.get("metric_topk") != group.protocol["topk"]
            or report.get("train_pairs_sha256") != group.protocol["train_pairs_sha256"]
            or report.get("candidate_protocol") != "full_ranking_mask_train_plus_valid"
            or report.get("graph_source") != "train_only"):
        raise ValueError(f"{path} 与当前验证选择或评价协议不一致")
    metrics = report["test_metrics"]
    if set(metrics) != {f"K{k}" for k in expected}:
        raise ValueError(f"{path} 的测试模型集合不正确")
    for values in metrics.values():
        for name in ("recall", "hit", "ndcg"):
            value = values[f"{name}@{group.protocol['topk']}"]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError(f"{path} 含非法测试指标")
    return metrics


def render_group(group: Group, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    topk = group.protocol["topk"]
    lr = group.protocol["learning_rate"]
    selected = sorted({0, group.selected_k})
    note = (f"lr = {lr:g}\nBest K = {group.selected_k}\n"
            "Selection: validation NDCG\n" +
            "\n".join(f"K{k}: best epoch {group.records[k]['best_epoch']}" for k in LAYERS))
    series = [(f"MF (K0)" if k == 0 else f"K{k}", group.histories[k],
               group.records[k]["best_epoch"]) for k in LAYERS]
    draw_curves(series, topk, report_dir / "k0_k4_curves", f"K0-K4 comparison | lr={lr:g}", note)
    draw_curves([series[k] for k in selected], topk, report_dir / "mf_vs_best_curves",
                f"MF vs validation-selected K={group.selected_k} | lr={lr:g}", note)
    all_rows = [group.table_row(k) for k in LAYERS]
    write_csv(report_dir / "best_epochs.csv", all_rows)
    comparison = [dict(all_rows[k]) for k in selected]
    labels = [f"MF (K0)\nepoch {group.records[k]['best_epoch']}" if k == 0 else
              f"Best K{k}\nepoch {group.records[k]['best_epoch']}" for k in selected]
    panels = [("train_ranking_loss", "Training BPR loss"),
              ("train_total_loss", "Training total loss"),
              ("train_pair_accuracy", "Training pair accuracy"),
              *[(f"valid_{m}@{topk}", f"Validation {m.upper()}@{topk}")
                for m in ("recall", "hit", "ndcg")]]
    draw_bars(comparison, labels, panels, report_dir / "mf_vs_best_at_selected_epochs",
              "Each model at its own best validation epoch | same learning rate")
    final_metrics = read_final_metrics(group)
    if final_metrics is not None:
        for row, k in zip(comparison, selected):
            row.update({f"test_{m}@{topk}": final_metrics[f"K{k}"][f"{m}@{topk}"]
                        for m in ("recall", "hit", "ndcg")})
        draw_bars(comparison, labels,
                  [(f"test_{m}@{topk}", f"Test {m.upper()}@{topk}")
                   for m in ("recall", "hit", "ndcg")],
                  report_dir / "mf_vs_best_test", "Saved final test metrics | mask Train + Valid")
    write_csv(report_dir / "mf_vs_best.csv", comparison)
    write_json(report_dir / "report.json", {
        "source_dir": str(group.directory.resolve()), "selected_k": group.selected_k,
        "selection_source": "validation_only", "shared_protocol": group.protocol,
        "layer_ablation": True, "controlled_variable": "n_layers",
        "test_metrics_available": final_metrics is not None,
        "loss_source": "training epoch average at each validation-selected epoch",
        "test_loss_available": False, "comparison": comparison,
    })
    (report_dir / "README.md").write_text(
        f"# Day 15 对比报告\n\n学习率：{lr:g}；验证集选中 K={group.selected_k}。\n\n"
        "- `k0_k4_curves.png`：五组训练 Loss、原始 L2、训练 pair accuracy 和验证 Recall/Hit/NDCG。\n"
        "- `mf_vs_best_curves.png`：同一学习率的 MF 与最佳 K 的逐轮比较。\n"
        "- `mf_vs_best_at_selected_epochs.png`：各自最佳验证轮次的训练 Loss 与验证指标。\n"
        "- `best_epochs.csv`：五组各自最佳轮次的数值和相对同学习率 K0 的增益。\n"
        "- `mf_vs_best.csv`：MF 与最佳 K 的数值对照。\n\n"
        + ("`mf_vs_best_test.png` 仅展示已有最终测试指标，没有重新评价。\n\n"
           if final_metrics is not None else "尚无最终测试记录，当前图表只展示训练与验证结果。\n\n")
        + "Loss 是训练时按样本加权的逐轮平均值，既不是验证/测试 Loss，"
        "也不是对已保存权重重新计算的 Loss。原始 L2 尚未乘 reg_lambda。\n\n"
        "固定 lr、初始化种子、训练轮数预算等配置比较 K0–K4，是传播层数消融；"
        "K0 去掉了图传播，是同实现的 MF 基线。若最优 K=0，对照只显示一个模型。\n",
        encoding="utf-8",
    )
    print(f"对比图与表格：{report_dir.resolve()}", flush=True)


def choose_sweep(groups: list[Group]) -> dict:
    """跨 lr 只改变 learning_rate；相同 NDCG 时先选较小 K，再选较小 lr。"""
    reference = groups[0].protocol
    for group in groups:
        for field in FAIR_FIELDS:
            if field != "learning_rate" and group.protocol[field] != reference[field]:
                raise ValueError(f"跨 lr 实验的 {field} 不一致，不能作为控制变量比较")
    group, record = max(
        ((g, r) for g in groups for r in g.records),
        key=lambda pair: (pair[1]["ndcg"], -pair[1]["n_layers"],
                          -pair[0].protocol["learning_rate"]),
    )
    return {
        "selection_source": "validation_only",
        "valid_metric": f"ndcg@{reference['topk']}",
        "selected_result_dir": group.directory.name,
        "selected_learning_rate": group.protocol["learning_rate"],
        "selected_k": record["n_layers"], "best_epoch": record["best_epoch"],
        "valid_ndcg": record["ndcg"],
        "tie_break": "smaller_n_layers_then_smaller_learning_rate",
        "mf_baseline": "K0 at the selected learning rate",
        "train_pairs_sha256": reference["train_pairs_sha256"],
        "candidate_grid": [{"learning_rate": g.protocol["learning_rate"],
                            "layers": list(LAYERS)} for g in groups],
    }


def lr_directory(lr: float) -> str:
    return f"lr_{lr}"


def load_sweep(directory: Path) -> tuple[list[Group], dict]:
    config = read_json(directory / "sweep_config.json")
    groups = [load_group(directory / lr_directory(lr)) for lr in config["lrs"]]
    for lr, group in zip(config["lrs"], groups):
        if group.protocol["learning_rate"] != lr:
            raise ValueError("学习率子目录与配置不一致")
        for field, value in config.items():
            if field in group.protocol and group.protocol[field] != value:
                raise ValueError(f"sweep_config.json 与实验的 {field} 不一致")
        for field, protocol_field in (("run_dir", "day7_run_dir"), ("graph_dir", "day11_graph_dir")):
            if config[field] != group.protocol[protocol_field]:
                raise ValueError(f"sweep_config.json 与实验的 {field} 不一致")
    choice = choose_sweep(groups)
    saved = directory / "lr_sweep_selection.json"
    if saved.exists() and read_json(saved) != choice:
        raise ValueError("当前验证记录与已冻结的 lr/K 选择不同，请检查实验目录")
    return groups, choice


def render_sweep(directory: Path, report_dir: Path) -> dict:
    groups, choice = load_sweep(directory)
    report_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        render_group(group, report_dir / group.directory.name)
    write_csv(report_dir / "lr_k_comparison.csv", [g.table_row(k) for g in groups for k in LAYERS])
    topk = groups[0].protocol["topk"]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(groups) * 0.65 + 2)))
    for ax, metric in zip(axes, ("recall", "hit", "ndcg")):
        values = [[r[metric] for r in group.records] for group in groups]
        image = ax.imshow(values, aspect="auto", cmap="Blues")
        ax.set(xticks=range(5), xticklabels=[f"K{k}" for k in LAYERS],
               yticks=range(len(groups)),
               yticklabels=[f"{g.protocol['learning_rate']:g}" for g in groups],
               xlabel="Propagation layers", ylabel="Learning rate",
               title=f"Validation {metric.upper()}@{topk}")
        threshold = (max(map(max, values)) + min(map(min, values))) / 2
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.4f}", ha="center", va="center",
                        color="white" if value > threshold else "#172554")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("LR x K sweep | all metrics at each run's best validation NDCG epoch")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, report_dir / "lr_k_heatmap", svg=True)
    draw_curves(
        [(f"lr={g.protocol['learning_rate']:g}, K{g.selected_k}",
          g.histories[g.selected_k], g.records[g.selected_k]["best_epoch"]) for g in groups],
        topk, report_dir / "lr_best_curves", "Best K within each learning rate",
        f"Selected lr = {choice['selected_learning_rate']:g}\n"
        f"Selected K = {choice['selected_k']}\nBest epoch = {choice['best_epoch']}\n"
        "All selection uses validation.\nMF baseline uses selected lr.",
    )
    write_json(report_dir / "lr_sweep_report.json", choice)
    print(f"验证集选择：lr={choice['selected_learning_rate']:g}, K={choice['selected_k']}, "
          f"epoch={choice['best_epoch']}, NDCG={choice['valid_ndcg']:.6f}", flush=True)
    return choice


def run_lr_sweep(args: argparse.Namespace) -> None:
    if (not args.lrs or any(not math.isfinite(lr) or lr <= 0 for lr in args.lrs)
            or len(set(args.lrs)) != len(args.lrs)):
        raise ValueError("--lrs 必须是互不重复的有限正数")
    if min(args.epochs, args.embedding_dim, args.batch_size, args.topk) <= 0:
        raise ValueError("epochs、embedding_dim、batch_size、topk 必须为正数")
    if not math.isfinite(args.reg_lambda) or args.reg_lambda < 0:
        raise ValueError("reg_lambda 必须是有限非负数")
    import torch
    from run_day15 import run_sweep

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    config = {name: getattr(args, name) for name in
              ("epochs", "embedding_dim", "batch_size", "reg_lambda", "topk", "seed")}
    config.update(lrs=sorted(args.lrs), device=device,
                  run_dir=str(args.run_dir.resolve()), graph_dir=str(args.graph_dir.resolve()))
    directory = args.output_dir.resolve()
    manifest = directory / "sweep_config.json"
    if manifest.exists():
        if read_json(manifest) != config:
            raise ValueError("已有 sweep 的搜索网格或配置不同，请使用新的 --output-dir")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("输出目录非空且没有 sweep_config.json，请使用新的 --output-dir")
    if any(directory.glob("lr_*/final_test_results.json")):
        raise ValueError("本 sweep 已做最终测试，不能继续搜索；已有结果可用 --stage plot 绘图")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(manifest, config)
    print(f"搜索 {len(config['lrs'])} 个 lr × 5 个 K，epochs={args.epochs}, device={device}", flush=True)
    for lr in config["lrs"]:
        run_args = argparse.Namespace(**vars(args))
        run_args.run_dir, run_args.graph_dir = config["run_dir"], config["graph_dir"]
        run_args.output_dir = str(directory / lr_directory(lr))
        run_args.learning_rate, run_args.device = lr, device
        run_sweep(run_args)
    _, choice = load_sweep(directory)
    write_json(directory / "lr_sweep_selection.json", choice)
    render_sweep(directory, args.report_dir or directory / "reports")
    print("sweep 完成；尚未计算测试指标。确认验证结果后，用 --stage final 做一次最终测试。", flush=True)


def run_sweep_final(args: argparse.Namespace) -> None:
    from run_day15 import run_final

    directory = args.output_dir.resolve()
    saved = read_json(directory / "lr_sweep_selection.json")
    _, choice = load_sweep(directory)
    if saved != choice:
        raise ValueError("最终评价必须使用已冻结的验证选择")
    config = read_json(directory / "sweep_config.json")
    if any(directory.glob("lr_*/final_test_results.json")):
        raise ValueError("这个 sweep 已有最终测试记录，请用 --stage plot 查看")
    run_args = argparse.Namespace(**config)
    run_args.output_dir = str(directory / choice["selected_result_dir"])
    run_args.device = config["device"] if args.device == "auto" else args.device
    run_final(run_args)
    render_sweep(directory, args.report_dir or directory / "reports")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("plot", "sweep", "final"), default="plot")
    parser.add_argument("--result-dir", type=Path, help="plot：原 Day 15 或多 lr sweep 的结果目录")
    parser.add_argument("--report-dir", type=Path, help="图表目录，默认结果目录下的 reports")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SWEEP, help="多 lr sweep 的独立输出根目录")
    parser.add_argument("--run-dir", type=Path, default=ROOT.parent / "week02-bpr" / "runs" / "day7_sgd_ml100k")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "experiments" / "day11_ml100k")
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.0003, 0.001, 0.003])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--reg-lambda", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.stage != "plot" and args.result_dir is not None:
        parser.error("sweep/final 用 --output-dir 指定搜索根目录；--result-dir 仅用于 plot")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    try:
        if args.stage == "sweep":
            run_lr_sweep(args)
        elif args.stage == "final":
            run_sweep_final(args)
        else:
            directory = args.result_dir or ROOT / "experiments" / "day15_ml100k"
            report_dir = args.report_dir or directory / "reports"
            if (directory / "sweep_config.json").is_file():
                render_sweep(directory, report_dir)
            else:
                render_group(load_group(directory), report_dir)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"实验工具错误：{error}\n")


if __name__ == "__main__":
    main()
