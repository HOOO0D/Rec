"""独立比较 ML-100K 上的 BPR-MF/MF 与 SGD/Adam，不修改已有实现。

运行示例：python -B plot_ml100k_comparison.py --epochs 30
所有 epoch 曲线使用验证集；按验证 AUC 选模型后，仅评估一次测试集。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

# 字体缓存放到临时目录，兼容用户目录只读的运行环境。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rec-learning-matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bpr_model import BPRMF
from model import MatrixFactorization, mf_loss
from negative_sampling import BootstrapTripletSampler, build_positive_sets
from train import load_ml100k_ratings, split_ratings, train_one_epoch

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def weight_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RankingEvaluator:
    """精确的逐用户宏平均 AUC、二元相关性 Top-K 指标及显式评分误差。

    候选集 = 当前留出正例 + 全数据中该用户从未观察的物品。
    另一留出集不作为负例；全部候选参与评估，不随机采负例。
    未观察物品是教学评估中的负例代理，不代表用户确实不喜欢。
    """
    def __init__(self, heldout, all_pairs, n_users, n_items, k, ratings):
        self.k = k
        self.n_items = n_items
        self.users = []
        known = build_positive_sets(all_pairs, n_users)
        positives = build_positive_sets(heldout, n_users)
        for u in range(n_users):
            pos = np.array(sorted(positives[u]), dtype=np.int64)
            neg = np.array([i for i in range(n_items) if i not in known[u]],
                           dtype=np.int64)
            if len(pos) and len(neg):
                # 排序后遇到同分时以物品 ID 升序确定 Top-K。
                candidates = np.sort(np.concatenate((pos, neg)))
                self.users.append((u, pos, neg, candidates))
        if not self.users:
            raise ValueError("留出集没有可同时评估正例和负例的用户")
        self.rating_users = np.array([u for u, i in heldout], dtype=np.int64)
        self.rating_items = np.array([i for u, i in heldout], dtype=np.int64)
        self.rating_values = np.array([ratings[u, i] for u, i in heldout])

    def from_scores(self, scores, explicit=False):
        if not np.isfinite(scores).all():
            raise ValueError("模型得分包含 NaN 或无穷值，请检查学习率")
        totals = np.zeros(5, dtype=np.float64)
        for u, pos, neg, candidates in self.users:
            ps = scores[u, pos]
            ns = np.sort(scores[u, neg])
            # 比正例分数低的负例计 1，同分负例计 0.5。
            left = np.searchsorted(ns, ps, side="left")
            right = np.searchsorted(ns, ps, side="right")
            auc = np.mean((left + right) / (2.0 * len(neg)))
            cutoff = min(self.k, len(candidates))
            order = np.argsort(-scores[u, candidates], kind="stable")[:cutoff]
            hits = np.isin(candidates[order], pos).astype(np.float64)
            discounts = 1.0 / np.log2(np.arange(cutoff) + 2)
            ideal = discounts[:min(len(pos), cutoff)].sum()
            totals += [auc, hits.sum() / len(pos), hits.sum() / cutoff,
                       float(hits @ discounts) / ideal, float(hits.any())]
        result = dict(zip(("auc", "recall", "precision", "ndcg", "hit_rate"),
                          (totals / len(self.users)).tolist()))
        result["evaluated_users"] = len(self.users)
        if explicit:
            error = scores[self.rating_users, self.rating_items] - self.rating_values
            result["rmse"] = float(np.sqrt(np.mean(error ** 2)))
            result["mae"] = float(np.mean(np.abs(error)))
        return result

    @torch.no_grad()
    def evaluate(self, model, explicit=False):
        model.eval()
        scores = model.user_embedding.weight @ model.item_embedding.weight.T
        return self.from_scores(scores.cpu().numpy(), explicit)


def train_mf_epoch(model, optimizer, data, batch_size, reg, generator):
    """沿用旧 MF 的已观察评分 MSE + 全表 L2，改为小批次更新。

    正则化仍调用已有 mf_loss；不将未观察值填零，不改成隐式目标。
    """
    model.train()
    order = torch.randperm(len(data), generator=generator)
    total = total_mse = 0.0
    for start in range(0, len(data), batch_size):
        batch = data[order[start:start + batch_size].to(data.device)]
        optimizer.zero_grad()
        loss = mf_loss(model, batch[:, 0].long(), batch[:, 1].long(),
                       batch[:, 2], reg_lambda=reg)
        with torch.no_grad():
            mse = loss - reg * model.l2_regularization()
        if not torch.isfinite(loss):
            raise ValueError("MF 损失非有限值，请降低学习率")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
        total_mse += mse.item() * len(batch)
    return {"loss": total / len(data), "data_loss": total_mse / len(data),
            "num_examples": len(data)}


def draw_curves(histories, out, k):
    colors = {"BPR-SGD": "#2563eb", "BPR-Adam": "#f97316",
              "MF-SGD": "#08916f", "MF-Adam": "#9333ea"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = [("auc", "AUC"), ("recall", f"Recall@{k}"),
               ("ndcg", f"NDCG@{k}"), ("precision", f"Precision@{k}"),
               ("hit_rate", f"HitRate@{k}"), ("rmse", "Rating RMSE (MF only)")]
    for ax, (key, title) in zip(axes.flat, metrics):
        for name, rows in histories.items():
            usable = [r for r in rows if key in r["validation"]]
            if usable:
                ax.plot([r["epoch"] for r in usable],
                        [r["validation"][key] for r in usable],
                        label=name, color=colors[name])
        ax.set(title=title, xlabel="Epoch (0 = initialization)", ylabel=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("MovieLens-100K | Validation | Same split and initial embeddings")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"validation_metrics.{suffix}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, family, objective in zip(axes, ("BPR", "MF"),
                                    ("Pairwise logistic loss", "Observed rating MSE")):
        for name, rows in histories.items():
            if name.startswith(family):
                usable = rows[1:]
                x = [r["epoch"] for r in usable]
                ax.plot(x, [r["loss"] for r in usable], label=f"{name}: total",
                        color=colors[name])
                ax.plot(x, [r["data_loss"] for r in usable], linestyle="--",
                        label=f"{name}: data only", color=colors[name], alpha=0.75)
        ax.set(title=f"{family}: {objective} + L2", xlabel="Epoch",
               ylabel="Training batch means, weighted by sample count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Different objectives: compare loss within each model family")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"training_loss.{suffix}", dpi=160)
    plt.close(fig)


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    data_path = args.data_path
    if data_path is None:
        data_path = next((p for p in (ROOT / "ml-100k/u.data", ROOT / "data/ml-100k/u.data")
                          if p.is_file()), ROOT / "ml-100k/u.data")
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到官方 ML-100K 数据，请指定 --data-path：{data_path}")
    records = load_ml100k_ratings(data_path)
    if (len(records), max(r[0] for r in records), max(r[1] for r in records)) != (100000, 942, 1681):
        raise ValueError("需要完整官方 ML-100K u.data：100000 条评分、943 用户、1682 物品")
    if len({(u, i) for u, i, _, _ in records}) != 100000:
        raise ValueError("数据中存在重复用户物品对")
    train, valid, test = split_ratings(records, args.valid_ratio, args.test_ratio, args.seed)
    rating_lookup = {(u, i): float(r) for u, i, r, _ in records}
    all_pairs = list(rating_lookup)
    n_users, n_items = 943, 1682
    valid_eval = RankingEvaluator(valid, all_pairs, n_users, n_items, args.k, rating_lookup)
    test_eval = RankingEvaluator(test, all_pairs, n_users, n_items, args.k, rating_lookup)
    out = args.output_dir
    # 防止不小心覆盖已有实验；重跑时换一个输出目录。
    out.mkdir(parents=True, exist_ok=False)
    for name, pairs in (("train", train), ("valid", valid), ("test", test)):
        (out / f"{name}_pairs.tsv").write_text(
            "".join(f"{u}\t{i}\n" for u, i in pairs), encoding="utf-8")
    torch.manual_seed(args.seed)
    initial_model = BPRMF(n_users, n_items, args.embedding_dim)
    initial = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}
    positives = build_positive_sets(train, n_users)
    sampler = BootstrapTripletSampler(positives, n_items)
    steps = args.steps_per_epoch or len(train)
    mf_data = torch.tensor([(u, i, rating_lookup[u, i]) for u, i in train],
                           dtype=torch.float32, device=args.device)
    config = {**vars(args), "data_path": str(data_path.resolve()),
              "output_dir": str(out.resolve()), "steps_per_epoch": steps,
              "n_users": n_users, "n_items": n_items,
              "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
              "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "initial_weight_sha256": weight_digest(initial),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("train.py", "bpr_model.py", "model.py", "negative_sampling.py")},
              "torch_version": str(torch.__version__),
              "selection": "best validation macro AUC, earliest epoch on ties",
              "relevance": "all observed ratings are implicit positives for ranking",
              "candidates": "heldout positives + all never-observed items per user",
              "auc": "exact per-user positive/negative pair probability; ties count 0.5; macro average",
              "topk_ties": "score descending, item ID ascending",
              "mf_training": "existing observed-rating MSE + full-table L2, shuffled minibatches",
              "bpr_training": "existing train_one_epoch, uniform legal triples with replacement",
              "note": "equal epochs are not equal wall time; no hyperparameter search; test evaluated only at selected epoch"}
    write_json(out / "config.json", config)
    histories, summary, bpr_hashes = {}, [], []
    specs = [("BPR-SGD", args.bpr_sgd_lr), ("BPR-Adam", args.bpr_adam_lr),
             ("MF-SGD", args.mf_sgd_lr), ("MF-Adam", args.mf_adam_lr)]
    for name, lr in specs:
        explicit = name.startswith("MF-")
        model = (MatrixFactorization if explicit else BPRMF)(n_users, n_items, args.embedding_dim)
        model.load_state_dict(initial)
        model.to(args.device)
        optimizer_cls = torch.optim.SGD if name.endswith("SGD") else torch.optim.Adam
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        rng = random.Random(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        directory = out / name
        directory.mkdir()
        rows = [{"epoch": 0, "validation": valid_eval.evaluate(model, explicit)}]
        best_auc, best_epoch, best_state = -math.inf, 0, None
        sample_hashes = []
        started = time.perf_counter()
        with (directory / "history.jsonl").open("w", encoding="utf-8") as log:
            log.write(json.dumps(rows[0], allow_nan=False) + "\n")
            for epoch in range(1, args.epochs + 1):
                if explicit:
                    stats = train_mf_epoch(model, optimizer, mf_data, args.batch_size,
                                           args.mf_reg, generator)
                else:
                    stats = train_one_epoch(model, sampler, optimizer, args.batch_size,
                                            steps, rng, args.bpr_l2, args.device)
                    stats["data_loss"] = stats["bpr_loss"]
                    sample_hashes.append(stats["sample_sha256"])
                metrics = valid_eval.evaluate(model, explicit)
                row = {"epoch": epoch, **stats, "validation": metrics}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                rows.append(row)
                if metrics["auc"] > best_auc:
                    best_auc, best_epoch = metrics["auc"], epoch
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}
                print(f"{name:8s} {epoch:3d}/{args.epochs} loss={stats['loss']:.5f} "
                      f"val_AUC={metrics['auc']:.4f} Recall@{args.k}={metrics['recall']:.4f}",
                      flush=True)
        if not explicit:
            bpr_hashes.append(sample_hashes)
        torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "config": {**config, "method": name, "lr": lr}}, directory / "final.pt")
        model.load_state_dict(best_state)
        test_metrics = test_eval.evaluate(model, explicit)
        torch.save({"state_dict": best_state, "epoch": best_epoch,
                    "config": {**config, "method": name, "lr": lr}}, directory / "best.pt")
        result = {"method": name, "lr": lr, "best_epoch": best_epoch,
                  "best_validation_auc": best_auc, "test": test_metrics,
                  "seconds": time.perf_counter() - started}
        write_json(directory / "result.json", result)
        summary.append(result)
        histories[name] = rows
    if bpr_hashes[0] != bpr_hashes[1]:
        raise AssertionError("两种 BPR 优化器的采样序列不一致")
    write_json(out / "summary.json", summary)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["method", "epoch", "loss", "data_loss", "auc", "recall", "precision",
                  "ndcg", "hit_rate", "rmse", "mae", "evaluated_users"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, rows in histories.items():
            for row in rows:
                writer.writerow({"method": name, "epoch": row["epoch"],
                                 "loss": row.get("loss"), "data_loss": row.get("data_loss"),
                                 **row["validation"]})
    draw_curves(histories, out, args.k)
    lines = ["# ML-100K 比较结果", "", "按验证集 AUC 选择 epoch，测试集只评估一次。",
             "", f"| 方法 | 最佳 epoch | 验证 AUC | 测试 AUC | Recall@{args.k} | NDCG@{args.k} |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        m = r["test"]
        lines.append(f"| {r['method']} | {r['best_epoch']} | {r['best_validation_auc']:.4f} | "
                     f"{m['auc']:.4f} | {m['recall']:.4f} | {m['ndcg']:.4f} |")
    lines += ["", "- validation_metrics.png/svg：验证集 AUC 和 Top-K 指标；RMSE 仅用于 MF。",
              "- training_loss.png/svg：两类目标分图，实线为含正则总损失，虚线为数据损失。",
              "- metrics.csv 和各方法 history.jsonl：逐轮数值，epoch 0 仅为初始化评估。",
              "- 四组采用相同数据划分、候选集和初始嵌入；两组 BPR 的逐轮采样摘要已核对一致。",
              "- MF 调用已有 mf_loss，但为匹配批次规模采用小批次训练；原 train_mf 是全批量。",
              "- BPR 学交互排序，MF 学 1～5 分评分：这是两种训练目标的比较。",
              "- 排序中所有已观察评分（包括低分）视为正例；未观察物品仅作为负例代理。",
              "- 验证候选排除训练和测试交互；测试候选排除训练和验证交互。",
              "- 指标为用户宏平均；AUC 精确比较全部正负对，同分计半；Top-K 同分按物品 ID 排序。",
              "- MF 的 L2 为全表平方和，BPR 为批内嵌入平方和；系数不能直接按数值比较。",
              "- 这是固定种子的教学基线，未搜索最佳超参数，不代表模型或优化器的普遍优劣。",
              "", "指标定义参考：",
              "- https://scikit-learn.org/stable/modules/generated/sklearn.metrics/roc_auc_score.html",
              "- https://scikit-learn.org/stable/modules/generated/sklearn.metrics/ndcg_score.html"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Finished: {out.resolve()}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/ml100k_comparison")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--bpr-sgd-lr", type=float, default=0.2)
    p.add_argument("--bpr-adam-lr", type=float, default=0.01)
    p.add_argument("--mf-sgd-lr", type=float, default=0.05)
    p.add_argument("--mf-adam-lr", type=float, default=0.01)
    p.add_argument("--bpr-l2", type=float, default=1e-4)
    p.add_argument("--mf-reg", type=float, default=1e-5)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.embedding_dim, args.k, args.threads) <= 0:
        p.error("epochs、batch-size、embedding-dim、k、threads 必须大于 0")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        p.error("steps-per-epoch 必须大于 0")
    if not (0 < args.valid_ratio < 1 and 0 < args.test_ratio < 1
            and args.valid_ratio + args.test_ratio < 1):
        p.error("验证、测试比例必须大于 0，且和小于 1")
    if any(not math.isfinite(x) or x <= 0 for x in
           (args.bpr_sgd_lr, args.bpr_adam_lr, args.mf_sgd_lr, args.mf_adam_lr)):
        p.error("学习率必须为有限正数")
    if any(not math.isfinite(x) or x < 0 for x in (args.bpr_l2, args.mf_reg)):
        p.error("正则系数必须为有限非负数")
    run(args)


if __name__ == "__main__":
    main()
