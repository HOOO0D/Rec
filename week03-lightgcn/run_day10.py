"""Day 10：把 Day 7 冻结的 MovieLens 训练交互转换成二部图。

使用方法：
    python run_day10.py --run-dir ../week02-bpr/runs/day7_sgd_ml100k \\
        --output-dir ./experiments/day10_ml100k

本脚手架可直接运行，但依赖你完成 graph_basics.py 的六个函数。
不读取原始 u.data，不使用验证/测试交互构建图。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from graph_basics import (
    build_bipartite_adjacency,
    build_interaction_matrix,
    node_degrees,
    verify_disjoint_splits,
)


def read_pairs(path: Path) -> list[tuple[int, int]]:
    """读取 Day 7 的零基用户—物品交互 TSV。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到交互文件：{path}")
    pairs = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path} 第 {line_number} 行应包含两个整数")
        try:
            pairs.append((int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise ValueError(f"{path} 第 {line_number} 行包含非整数编号") from exc
    return pairs


def run_day10(run_dir: str | Path, output_dir: str | Path) -> dict:
    """从已经冻结的数据划分构造训练图并保存必要实验元数据。"""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    conf_file = run_dir / "config.json"
    if not conf_file.is_file():
        raise FileNotFoundError(f"找不到 Day 7 配置：{conf_file}")
    config = json.loads(conf_file.read_text(encoding="utf-8"))
    num_users = int(config["n_users"])
    num_items = int(config["n_items"])
    train = read_pairs(run_dir / "train_pairs.tsv")
    valid = read_pairs(run_dir / "valid_pairs.tsv")
    test = read_pairs(run_dir / "test_pairs.tsv")

    # 验证、测试交互只用于检查划分，绝不传给图构造函数。
    sizes = verify_disjoint_splits(train, valid, test)
    for name in ("n_train", "n_valid", "n_test"):
        if sizes[name] != int(config[name]):
            raise ValueError(f"配置记录与冻结的 {name} 不一致")
    for u, i in train + valid + test:
        if not (0 <= u < num_users and 0 <= i < num_items):
            raise ValueError(f"用户或物品编号越界：({u}, {i})")

    # 稠密交互矩阵可用于小数据验收；真正用于 Day 11 的是稀疏图。
    r = build_interaction_matrix(train, num_users, num_items)
    adjacency = build_bipartite_adjacency(train, num_users, num_items)
    adjacency = adjacency.coalesce()
    degrees = node_degrees(adjacency)

    if adjacency.layout != torch.sparse_coo or not adjacency.is_coalesced():
        raise ValueError("邻接矩阵必须是已合并的 COO 稀疏张量")
    if adjacency.shape != (num_users + num_items, num_users + num_items):
        raise ValueError("邻接矩阵形状不正确")
    if not torch.equal(r, adjacency.to_dense()[:num_users, num_users:]):
        raise ValueError("邻接矩阵的交互块与训练交互矩阵不一致")
    edge_count = len(train)
    if adjacency._nnz() != 2 * edge_count:
        raise ValueError("无向边必须存储两个方向，且不能存在重复边")
    if int(degrees.sum().item()) != 2 * edge_count:
        raise ValueError("节点度之和应为训练无向边数的两倍")

    # 记录训练交互的稳定摘要，以便 Day 11–15 核对图源。
    digest = hashlib.sha256()
    for u, i in sorted(train):
        digest.update(f"{u}\t{i}\n".encode("utf-8"))
    summary = {
        "num_users": num_users,
        "num_items": num_items,
        "num_nodes": num_users + num_items,
        "num_train_pairs": edge_count,
        "num_undirected_edges": edge_count,
        "num_stored_directed_entries": int(adjacency._nnz()),
        "num_isolated_nodes": int((degrees == 0).sum().item()),
        "num_valid_pairs_audited": len(valid),
        "num_test_pairs_audited": len(test),
        "adjacency_shape": list(adjacency.shape),
        "adjacency_layout": "sparse_coo",
        "graph_source": "train_only",
        "has_self_loops": False,
        "has_normalization": False,
        "train_pairs_sha256": digest.hexdigest(),
        "day7_seed": config.get("seed"),
        "day7_run_dir": str(run_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adjacency, output_dir / "adjacency.pt")
    torch.save(degrees, output_dir / "degrees.pt")
    (output_dir / "graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 10：构造 LightGCN 的训练交互二部图")
    parser.add_argument("--run-dir", required=True, help="Day 7 冻结的训练实验目录")
    parser.add_argument("--output-dir", required=True, help="Day 10 图输出目录")
    args = parser.parse_args()
    info = run_day10(args.run_dir, args.output_dir)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
