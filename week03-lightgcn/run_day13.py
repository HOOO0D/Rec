"""Day 13：读取 Day 11 的训练图，做一次 LightGCN 前向并保存检查产物。

此程序不训练模型，不读取原始 MovieLens，不重新划分数据。
需要先完成 lightgcn_model.py 中所有 TODO。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from lightgcn_model import LightGCN


def run_day13(graph_dir: str | Path, output_dir: str | Path, *, embedding_dim: int = 16,
              n_layers: int = 2, seed: int = 20261031, device: str = "cpu") -> dict:
    graph_dir = Path(graph_dir)
    output_dir = Path(output_dir)
    summary_path = graph_dir / "normalization_summary.json"
    adjacency_path = graph_dir / "normalized_adjacency.pt"
    for path in (summary_path, adjacency_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 Day 11 文件：{path}")

    source = json.loads(summary_path.read_text(encoding="utf-8"))
    if source.get("graph_source") != "train_only":
        raise ValueError("只接受由训练交互构建的图：graph_source 必须为 train_only")
    if source.get("normalization") != "symmetric_degree_no_self_loops":
        raise ValueError("需要 Day 11 的对称归一化且不含自环的图")
    n_users, n_items = int(source["num_users"]), int(source["num_items"])
    n_nodes = n_users + n_items
    if n_users < 1 or n_items < 1:
        raise ValueError("用户和物品数量应为正数")

    # Day 11 的产物是单个张量而不是任意对象；只加载权重/张量类型。
    adjacency = torch.load(adjacency_path, map_location="cpu", weights_only=True)
    if adjacency.layout != torch.sparse_coo or tuple(adjacency.shape) != (n_nodes, n_nodes):
        raise ValueError("归一化邻接矩阵必须为 [U+I,U+I] 的 COO 稀疏张量")
    adjacency = adjacency.coalesce()
    if adjacency.dtype != torch.float32 or not torch.isfinite(adjacency.values()).all():
        raise ValueError("归一化邻接矩阵必须为有限的 float32 数值")
    if torch.any(adjacency.indices()[0] == adjacency.indices()[1]):
        raise ValueError("LightGCN 的传播邻接矩阵不得添加自环")
    if int(adjacency._nnz()) != int(source["num_sparse_entries"]):
        raise ValueError("邻接矩阵边数与 Day 11 的归一化摘要不符")

    # 显式初始化随机种子，检查同一配置重复运行的结果可复现。
    torch.manual_seed(seed)
    model = LightGCN(n_users=n_users, n_items=n_items, embedding_dim=embedding_dim,
                     n_layers=n_layers, normalized_adjacency=adjacency).to(device)
    model.eval()
    with torch.no_grad():
        users, items = model()
        b = min(5, n_users)
        sample_users = torch.arange(b, dtype=torch.long, device=users.device)
        sample_items = torch.zeros(b, dtype=torch.long, device=users.device)
        pair_scores = model.score(sample_users, sample_items)
        rank_scores = model.full_sort_scores(sample_users)
    if users.shape != (n_users, embedding_dim) or items.shape != (n_items, embedding_dim):
        raise ValueError("最终用户/物品 Embedding 形状错误")
    if rank_scores.shape != (b, n_items):
        raise ValueError("全物品评分形状错误")
    if not (torch.isfinite(users).all() and torch.isfinite(items).all() and torch.isfinite(rank_scores).all()):
        raise ValueError("模型前向出现 NaN/Inf")
    if not torch.allclose(pair_scores, rank_scores[:, 0], atol=1e-6):
        raise ValueError("score() 与 full_sort_scores() 的输出不一致")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_params = (n_users + n_items) * embedding_dim
    if param_count != expected_params:
        raise ValueError(f"LightGCN 不应引入额外可训练参数：{param_count} != {expected_params}")
    report = {
        "graph_source": source["graph_source"],
        "normalization": source["normalization"],
        "train_pairs_sha256": source.get("train_pairs_sha256"),
        "num_users": n_users,
        "num_items": n_items,
        "num_nodes": n_nodes,
        "num_sparse_entries": int(adjacency._nnz()),
        "embedding_dim": embedding_dim,
        "n_layers": n_layers,
        "seed": seed,
        "device": str(users.device),
        "n_trainable_parameters": param_count,
        "n_expected_trainable_parameters": expected_params,
        "user_embedding_shape": list(users.shape),
        "item_embedding_shape": list(items.shape),
        "full_sort_shape": list(rank_scores.shape),
        "is_trained": False,
        "note": "仅做 Day 13 初始模型前向和数值审计；没有优化器与 BPR 训练",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "forward_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({
        "user_ids": sample_users.cpu(),
        "item_ids": sample_items.cpu(),
        "paired_scores": pair_scores.cpu(),
        "full_sort_scores": rank_scores.cpu(),
        "user_embeddings_sample": users[:b].cpu(),
        "item_embeddings_sample": items[:min(5, n_items)].cpu(),
    }, output_dir / "forward_sample.pt")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 13：LightGCN 多层传播和前向集成")
    parser.add_argument("--graph-dir", required=True, help="Day 11 的归一化图实验目录")
    parser.add_argument("--output-dir", required=True, help="Day 13 输出目录")
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_day13(args.graph_dir, args.output_dir,
                                embedding_dim=args.embedding_dim,
                                n_layers=args.n_layers, seed=args.seed,
                                device=args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
