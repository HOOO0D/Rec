"""Day 14：读取 Day 7 冻结数据和 Day 11 图，训练并验证 LightGCN。

依赖：先完成 Day 13 lightgcn_model.py，再完成 Day 14 lightgcn_train.py 的 TODO。
只从 Day 7 的训练集生成负样本和图；验证集仅参与指标计算；测试集仅做划分不重叠审计。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import csv
import torch

from lightgcn_model import LightGCN
from lightgcn_train import build_seen_sets, train_one_epoch, evaluate_full_ranking


def read_pairs(path: Path) -> list[tuple[int, int]]:
    """读取 Day 7 冻结的无表头 TSV（每行两个非负整数），禁止重复交互。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到交互文件：{path}")
    pairs = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path} 第 {lineno} 行格式不是 '用户ID\\t物品ID'")
        try:
            u, i = map(int, fields)
        except ValueError as exc:
            raise ValueError(f"{path} 第 {lineno} 行必须是整数 ID") from exc
        if u < 0 or i < 0:
            raise ValueError(f"{path} 第 {lineno} 行不能出现负 ID")
        pairs.append((u, i))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{path} 包含重复 (u,i)")
    return pairs


def pair_hash(pairs: list[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for u, i in sorted(pairs):
        digest.update(f"{u}\t{i}\n".encode("utf-8"))
    return digest.hexdigest()


def load_and_audit(run_dir: Path, graph_dir: Path):
    """核对真实训练集合与图中每一条有向边，避免图和训练数据版本不一致。"""
    config_path = run_dir / "config.json"
    graph_summary_path = graph_dir / "normalization_summary.json"
    graph_path = graph_dir / "normalized_adjacency.pt"
    for path in (config_path, graph_summary_path, graph_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 Day 7/Day 11 文件：{path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(graph_summary_path.read_text(encoding="utf-8"))
    train = read_pairs(run_dir / "train_pairs.tsv")
    valid = read_pairs(run_dir / "valid_pairs.tsv")
    test = read_pairs(run_dir / "test_pairs.tsv")
    if not train or not valid:
        raise ValueError("至少需要非空的训练集和验证集")
    sets = [set(train), set(valid), set(test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("训练/验证/测试交互存在重叠，拒绝继续")
    users, items = int(config["n_users"]), int(config["n_items"])
    if users <= 0 or items <= 1:
        raise ValueError("n_users、n_items 参数不合法")
    for name, pairs in (("train", train), ("valid", valid), ("test", test)):
        for u, i in pairs:
            if not 0 <= u < users or not 0 <= i < items:
                raise ValueError(f"{name} 中出现越界交互：({u}, {i})")
    for key, pairs in (("n_train",train), ("n_valid",valid), ("n_test",test)):
        if key in config and int(config[key]) != len(pairs):
            raise ValueError(f"Day 7 的 {key} 与实际文件数量不一致")
    if summary.get("graph_source") != "train_only" or summary.get("normalization") != "symmetric_degree_no_self_loops":
        raise ValueError("Day 11 图不是训练专用、无自环的对称归一化图")
    if (int(summary.get("num_users", -1)), int(summary.get("num_items", -1))) != (users, items):
        raise ValueError("Day 11 图节点数与 Day 7 冻结数据不一致")
    digest = pair_hash(train)
    if summary.get("train_pairs_sha256") != digest:
        raise ValueError("Day 11 图源 SHA256 与 Day 7 的 train_pairs.tsv 不匹配")
    adj = torch.load(graph_path, map_location="cpu", weights_only=True)
    n_nodes = users + items
    if adj.layout != torch.sparse_coo or adj.dtype != torch.float32 or adj.shape != (n_nodes, n_nodes):
        raise ValueError("Day 11 归一化邻接矩阵应为正确尺寸的 float32 稀疏 COO")
    adj = adj.coalesce()
    rows, cols = adj.indices()[0], adj.indices()[1]
    values = adj.values()
    if not torch.isfinite(values).all() or (values <= 0).any() or (rows == cols).any():
        raise ValueError("图存在不合法边权或自环")
    if adj._nnz() != 2 * len(train) or adj._nnz() != int(summary["num_sparse_entries"]):
        raise ValueError("图中边数与训练交互数不匹配")
    # 不只检查摘要：直接核对全部图边与训练交互完全一致，防止验证/测试边泄漏。
    u = torch.tensor([p[0] for p in train], dtype=torch.long)
    i = torch.tensor([p[1] for p in train], dtype=torch.long) + users
    exp_rows = torch.cat([u, i])
    exp_cols = torch.cat([i, u])
    actual_keys = torch.sort(rows * n_nodes + cols).values
    expected_keys = torch.sort(exp_rows * n_nodes + exp_cols).values
    if not torch.equal(actual_keys, expected_keys):
        raise ValueError("图中出现非训练边、缺失训练边或错误全局节点编号")
    # 复核每条边的对称归一化权重。
    deg = torch.bincount(exp_rows, minlength=n_nodes).to(torch.float32)
    expected_weights = (deg[rows] * deg[cols]).rsqrt()
    if not torch.allclose(values, expected_weights, atol=1e-6, rtol=1e-5):
        raise ValueError("图的边权与对称归一化权重不一致")
    return config, summary, train, valid, test, adj, digest


def run_day14(
    run_dir: str | Path,
    graph_dir: str | Path,
    output_dir: str | Path,
    *,
    embedding_dim: int = 16,
    n_layers: int = 2,
    batch_size: int = 1024,
    epochs: int = 10,
    learning_rate: float = 0.001,
    reg_lambda: float = 1e-4,
    topk: int = 10,
    seed: int = 20261031,
    device: str = "cpu",
    eval_user_batch_size: int = 128,
) -> dict:
    """运行 Day 14，生成训练记录与最佳验证模型；不计算测试集最终指标。"""
    run_dir, graph_dir, output_dir = Path(run_dir), Path(graph_dir), Path(output_dir)
    if batch_size <= 0 or epochs <= 0 or embedding_dim <= 0 or n_layers < 0 or topk <= 0:
        raise ValueError("批量、轮数、维度、层数或 K 不合法")
    if learning_rate <= 0 or not math.isfinite(learning_rate) or reg_lambda < 0 or not math.isfinite(reg_lambda):
        raise ValueError("学习率或正则化系数不合法")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无可用 CUDA；请改为 --device cpu")
    config, source, train, valid, test, adj, digest = load_and_audit(run_dir, graph_dir)
    n_users, n_items = int(config["n_users"]), int(config["n_items"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    model = LightGCN(n_users, n_items, embedding_dim, n_layers, adj).to(device)
    seen = build_seen_sets(train, n_users, n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_log: list[dict] = []
    best_ndcg, best_epoch, best_valid = float("-inf"), 0, None
    checkpoint = output_dir / "best_valid_model.pt"
    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, train, seen, optimizer, batch_size=batch_size,
                                reg_lambda=reg_lambda, rng=rng)
        metrics = evaluate_full_ranking(model, train, valid, k=topk,
                                        eval_user_batch_size=eval_user_batch_size)
        if not all(math.isfinite(float(v)) for v in stats.values() if isinstance(v, (int, float))):
            raise ValueError("训练指标存在 NaN 或 Inf")
        if not all(math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float))):
            raise ValueError("验证指标存在 NaN 或 Inf")
        row = {"epoch": epoch, **stats, **metrics}
        result_log.append(row)
        key = f"ndcg@{topk}"
        value = float(metrics[key])
        if value > best_ndcg:
            best_ndcg, best_epoch, best_valid = value, epoch, metrics
            # 图作为 Day 13 模型的 buffer 一并保存，方便载入后执行相同的传播。
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch, "best_valid": metrics, "seed": seed,
                        "graph_sha256": digest,
                        "model_config": {"n_users": n_users, "n_items": n_items,
                                         "embedding_dim": embedding_dim, "n_layers": n_layers}}, checkpoint)
        print(f"Epoch {epoch:03d}/{epochs:03d} | BPR={stats['ranking_loss']:.5f} | "
              f"验证 Recall@{topk}={metrics[f'recall@{topk}']:.4f} "
              f"NDCG@{topk}={metrics[key]:.4f}", flush=True)
    with (output_dir / "epoch_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_log[0]))
        writer.writeheader()
        writer.writerows(result_log)
    save_config = {"embedding_dim": embedding_dim, "n_layers": n_layers,
                   "batch_size": batch_size, "epochs": epochs, "learning_rate": learning_rate,
                   "reg_lambda": reg_lambda, "topk": topk, "seed": seed, "device": device,
                   "optimizer": "Adam", "weight_decay": 0.0,
                   "negative_sampling": "train_unseen_uniform_one_per_positive",
                   "l2_convention": "initial_triplet_rows_sum_sq_div_2B",
                   "eval": "valid_full_ranking_macro_users_mask_train",
                   "train_pairs_sha256": digest,
                   "day7_run_dir": str(run_dir), "day11_graph_dir": str(graph_dir)}
    (output_dir / "experiment_config.json").write_text(
        json.dumps(save_config, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"graph_source": "train_only", "train_pairs_sha256": digest,
              "n_users": n_users, "n_items": n_items,
              "n_train": len(train), "n_valid": len(valid), "n_test_audited_not_evaluated": len(test),
              "best_epoch": best_epoch, "best_valid_result": best_valid,
              "checkpoint": str(checkpoint), "training_epochs": epochs,
              "training_last_epoch": result_log[-1], "test_result": None,
              "note": "Day 14 仅训练和验证，不依据测试集选模型；Day 15 进行公平对照与最终测试。"}
    (output_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 14｜LightGCN BPR 训练与 Full Ranking 验证")
    parser.add_argument("--run-dir", required=True, help="Day 7 的冻结划分目录")
    parser.add_argument("--graph-dir", required=True, help="Day 11 归一化训练图目录")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reg-lambda", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    result = run_day14(args.run_dir, args.graph_dir, args.output_dir,
                       embedding_dim=args.embedding_dim, n_layers=args.n_layers,
                       batch_size=args.batch_size, epochs=args.epochs,
                       learning_rate=args.learning_rate, reg_lambda=args.reg_lambda,
                       topk=args.topk, seed=args.seed, device=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
