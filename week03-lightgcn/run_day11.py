"""Day 11：使用 Day 10 冻结的训练图，生成归一化邻接矩阵和单层聚合检查。

示例命令：
    python run_day11.py --graph-dir experiments/day10_ml100k \\
        --output-dir experiments/day11_ml100k

依赖你独立完成 graph_aggregation.py 的五个函数。
本文件不重新读取评分数据，不重新划分交互，也不训练推荐模型。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from graph_aggregation import (
    degrees_from_adjacency,
    inverse_sqrt_degrees,
    normalize_bipartite_adjacency,
    sparse_aggregate,
)


def run_day11(graph_dir: str | Path, output_dir: str | Path) -> dict:
    """读取原始训练图，核对来源与度数，计算归一化和一跳聚合并保存。"""
    graph_dir = Path(graph_dir)
    output_dir = Path(output_dir)
    graph_summary_path = graph_dir / "graph_summary.json"
    graph_path = graph_dir / "adjacency.pt"
    degree_path = graph_dir / "degrees.pt"
    for path in (graph_summary_path, graph_path, degree_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 Day 10 图实验产物：{path}")

    original_summary = json.loads(graph_summary_path.read_text(encoding="utf-8"))
    # 严格区分图源：训练集建图，验证/测试集只用于上游的交叉检查。
    if original_summary.get("graph_source") != "train_only":
        raise ValueError("输入图来源必须是 train_only")
    if original_summary.get("has_self_loops") is not False:
        raise ValueError("Day 11 使用无自环二部图，不允许输入含自环的图")
    if original_summary.get("has_normalization") is not False:
        raise ValueError("需要输入 Day 10 原始未归一化图")
    n_users = int(original_summary["num_users"])
    n_items = int(original_summary["num_items"])
    n_nodes = n_users + n_items

    # 仅载入张量，不加载任意 Python 对象。
    adjacency = torch.load(graph_path, map_location="cpu", weights_only=True).coalesce()
    saved_degrees = torch.load(degree_path, map_location="cpu", weights_only=True)
    if adjacency.layout != torch.sparse_coo or tuple(adjacency.shape) != (n_nodes, n_nodes):
        raise ValueError("输入图必须是 [U+I,U+I] 的稀疏 COO 方阵")
    if adjacency._nnz() != int(original_summary["num_stored_directed_entries"]):
        raise ValueError("图的非零边数量与 Day 10 摘要不一致")
    if adjacency._nnz() != 2 * int(original_summary["num_train_pairs"]):
        raise ValueError("原始二部图的双向边数量不符合预期")
    if adjacency._nnz() and torch.any(adjacency.indices()[0] == adjacency.indices()[1]):
        raise ValueError("原始图意外包含自环")
    degrees = degrees_from_adjacency(adjacency)
    if not torch.allclose(degrees, saved_degrees.to(torch.float32)):
        raise ValueError("重新计算的节点度与 Day 10 保存的度数不一致")
    if not torch.allclose(adjacency.values(), torch.ones_like(adjacency.values())):
        raise ValueError("本次要求输入 Day 10 二值图，而不是提前加权的图")
    inv_degrees = inverse_sqrt_degrees(degrees)
    normalized = normalize_bipartite_adjacency(adjacency)
    if normalized.layout != torch.sparse_coo or not normalized.is_coalesced():
        raise ValueError("归一化结果必须是已合并的 COO 稀疏张量")
    if tuple(normalized.shape) != (n_nodes, n_nodes):
        raise ValueError("归一化矩阵形状改变")
    if not torch.equal(normalized.indices(), adjacency.indices()):
        raise ValueError("归一化不得新增或删除训练边")
    if not torch.isfinite(normalized.values()).all():
        raise ValueError("归一化边权出现 NaN/Inf")

    # 构造确定性的测试用节点特征，只检查传播数值，不当作训练得到的 Embedding。
    ids = torch.arange(n_nodes, dtype=torch.float32)
    features = torch.stack((ids / max(1, n_nodes), torch.ones_like(ids), ids.remainder(5)), dim=1)
    one_hop = sparse_aggregate(normalized, features)
    if tuple(one_hop.shape) != tuple(features.shape) or not torch.isfinite(one_hop).all():
        raise ValueError("单层稀疏聚合输出形状或数值错误")
    if (degrees == 0).any() and not torch.allclose(one_hop[degrees == 0], torch.zeros_like(one_hop[degrees == 0])):
        raise ValueError("无自环孤立节点的一层聚合应为零向量")

    report = {
        "num_users": n_users,
        "num_items": n_items,
        "num_nodes": n_nodes,
        "num_train_undirected_edges": int(original_summary["num_train_pairs"]),
        "num_sparse_entries": int(normalized._nnz()),
        "num_isolated_nodes": int((degrees == 0).sum().item()),
        "normalization": "symmetric_degree_no_self_loops",
        "graph_source": "train_only",
        "train_pairs_sha256": original_summary.get("train_pairs_sha256"),
        "day7_seed": original_summary.get("day7_seed"),
        "day10_graph_dir": str(graph_dir),
        "features_note": "仅为一次传播的确定性数值检查，不是可训练用户物品 Embedding",
        "all_edge_weights_finite": bool(torch.isfinite(normalized.values()).all().item()),
        "all_aggregate_values_finite": bool(torch.isfinite(one_hop).all().item()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(normalized, output_dir / "normalized_adjacency.pt")
    torch.save(inv_degrees, output_dir / "inverse_sqrt_degrees.pt")
    torch.save({
        "initial": features[:5].clone(),
        "one_hop": one_hop[:5].clone(),
    }, output_dir / "one_hop_sample.pt")
    (output_dir / "normalization_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 11：LightGCN 邻居聚合与稀疏归一化")
    parser.add_argument("--graph-dir", required=True, help="Day 10 的图实验目录")
    parser.add_argument("--output-dir", required=True, help="Day 11 的输出目录")
    args = parser.parse_args()
    print(json.dumps(run_day11(args.graph_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
