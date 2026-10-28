"""Day 10｜用户—物品二部图自动验收。

普通运行：python Test_Day10.py
完整验收：python Test_Day10.py --run-dir E:\\AI\\rec-learning\\week02-bpr\\runs\\day7_sgd_ml100k

A–E 使用固定合成数据，合计 80 分。
F 检查真实 Day 7 MovieLens-100K 冻结划分和 Day 10 图构造，20 分。
未提供 Day 7 产物时，不能宣称已完成 F 项。
"""
from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from pathlib import Path

import torch

from graph_basics import (
    item_global_id,
    build_interaction_matrix,
    build_bipartite_adjacency,
    node_degrees,
    neighbors_of,
    verify_disjoint_splits,
)

# 训练交互共四条，覆盖两个用户、三个物品。
TRAIN = [(0, 0), (0, 2), (1, 0), (1, 1)]
VALID = [(0, 1)]
TEST = [(1, 2)]
EXPECTED_R = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.float32)
EXPECTED_A = torch.tensor([
    [0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
], dtype=torch.float32)


def ensure(condition, explanation):
    """断言测试条件。"""
    if not bool(condition):
        raise AssertionError(explanation)


def must_raise_value_error(fn, context):
    """检查非法输入是否会抛出可诊断的异常。"""
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"{context}：应抛出 ValueError")


def test_a_mapping():
    """A：用户与物品节点位于不同的全局编号区间。"""
    ensure(item_global_id(0, 2, 3) == 2, "I0 的节点编号应为 2")
    ensure(item_global_id(1, 2, 3) == 3, "I1 的节点编号应为 3")
    ensure(item_global_id(2, 2, 3) == 4, "I2 的节点编号应为 4")
    ensure(item_global_id(0, 943, 1682) == 943, "MovieLens 的 I0 应位于全局编号 943")
    must_raise_value_error(lambda: item_global_id(-1, 2, 3), "负物品编号")
    must_raise_value_error(lambda: item_global_id(3, 2, 3), "物品编号越界")
    must_raise_value_error(lambda: item_global_id(0, -1, 3), "用户数量非法")
    must_raise_value_error(lambda: item_global_id(0, 2, -1), "物品数量非法")


def test_b_interactions():
    """B：交互矩阵形状和取值必须与手算一致。"""
    r = build_interaction_matrix(TRAIN, 2, 3)
    ensure(isinstance(r, torch.Tensor), "R 必须返回 Tensor")
    ensure(r.layout == torch.strided, "本日 R 使用稠密矩阵，用于手算对照")
    ensure(r.dtype == torch.float32, "R 的类型应为 float32")
    ensure(r.shape == (2, 3), "R 应为 2×3")
    ensure(torch.equal(r, EXPECTED_R), "R 与示例手算矩阵不一致")
    r2 = build_interaction_matrix(TRAIN + [(0, 0), (1, 1)], 2, 3)
    ensure(torch.equal(r2, EXPECTED_R), "重复交互仍然是二值 1，不能累加")
    r_empty = build_interaction_matrix([], 3, 4)
    ensure(r_empty.shape == (3, 4) and torch.count_nonzero(r_empty).item() == 0,
           "空训练数据应返回指定形状的零矩阵")
    must_raise_value_error(lambda: build_interaction_matrix([(2, 0)], 2, 3), "用户越界")
    must_raise_value_error(lambda: build_interaction_matrix([(0, 3)], 2, 3), "物品越界")


def test_c_sparse_adjacency():
    """C：无向二部图使用双向稀疏 COO，必须有对称的块结构。"""
    a = build_bipartite_adjacency(TRAIN, 2, 3)
    ensure(isinstance(a, torch.Tensor), "A 必须返回 Tensor")
    ensure(a.layout == torch.sparse_coo, "A 必须是稀疏 COO 格式")
    ensure(a.is_coalesced(), "应调用 coalesce() 合并重复坐标")
    ensure(a.dtype == torch.float32, "A 的边权类型必须为 float32")
    ensure(a.shape == (5, 5), "邻接矩阵应为 (U+I)×(U+I) = 5×5")
    ensure(a._nnz() == 8, "4 条无向交互边必须存储 8 个双向条目")
    ensure(torch.equal(a.to_dense(), EXPECTED_A), "A 与手算邻接矩阵不一致")
    ensure(torch.equal(a.to_dense(), a.to_dense().T), "无向图邻接矩阵应对称")
    ensure(torch.count_nonzero(a.to_dense()[:2, :2]).item() == 0, "不允许用户—用户边")
    ensure(torch.count_nonzero(a.to_dense()[2:, 2:]).item() == 0, "不允许物品—物品边")
    ensure(torch.count_nonzero(torch.diag(a.to_dense())).item() == 0, "不允许自环")
    ensure(torch.all(a.values() == 1).item(), "所有二值边权必须为 1")


def test_d_degree_neighbors():
    """D：节点度、邻居和握手定理。"""
    a = build_bipartite_adjacency(TRAIN, 2, 3)
    deg = node_degrees(a)
    ensure(isinstance(deg, torch.Tensor), "节点度必须返回 Tensor")
    ensure(deg.dtype == torch.int64, "节点度必须为 int64")
    ensure(deg.shape == (5,), "必须返回每个节点的度，包括孤立节点")
    ensure(deg.tolist() == [2, 2, 2, 1, 1], "节点度手算应是 [2,2,2,1,1]")
    ensure(int(deg.sum().item()) == 8, "度之和必须等于无向边数的两倍")
    for node_id, expected in enumerate([[2, 4], [2, 3], [0, 1], [1], [0]]):
        ensure(neighbors_of(a, node_id) == expected, f"全局节点 {node_id} 的邻居错误")
    must_raise_value_error(lambda: neighbors_of(a, -1), "负节点编号")
    must_raise_value_error(lambda: neighbors_of(a, 5), "节点编号越界")


def test_e_boundaries_leakage():
    """E：孤立节点、重复边和数据泄漏审计。"""
    a = build_bipartite_adjacency([(0, 1), (0, 1)], 3, 4)
    ensure(a.shape == (7, 7) and a._nnz() == 2, "重复交互只应构成一条无向边")
    ensure(node_degrees(a).tolist() == [1, 0, 0, 0, 1, 0, 0], "应保留五个孤立节点")
    ensure(neighbors_of(a, 2) == [], "孤立用户没有邻居")
    ensure(neighbors_of(a, 6) == [], "孤立物品没有邻居")
    empty = build_bipartite_adjacency([], 2, 3)
    ensure(empty.shape == (5, 5) and empty._nnz() == 0, "空图必须是 5×5 稀疏零矩阵")
    ensure(node_degrees(empty).tolist() == [0] * 5, "空图应保留全部孤立节点")
    must_raise_value_error(lambda: build_bipartite_adjacency([(2, 1)], 2, 3), "训练用户越界")
    must_raise_value_error(lambda: build_bipartite_adjacency([(0, 3)], 2, 3), "训练物品越界")
    checks = verify_disjoint_splits(TRAIN, VALID, TEST)
    ensure(checks == {"n_train": 4, "n_valid": 1, "n_test": 1}, "划分数量错误")
    must_raise_value_error(lambda: verify_disjoint_splits(TRAIN + [(0, 0)], VALID, TEST), "训练集重复")
    must_raise_value_error(lambda: verify_disjoint_splits(TRAIN, VALID + [(0, 1)], TEST), "验证集重复")
    must_raise_value_error(lambda: verify_disjoint_splits(TRAIN, VALID, TEST + [(1, 2)]), "测试集重复")
    must_raise_value_error(lambda: verify_disjoint_splits(TRAIN, VALID + [(0, 0)], TEST), "训练/验证交叉")
    must_raise_value_error(lambda: verify_disjoint_splits(TRAIN, VALID, TEST + [(0, 0)]), "训练/测试交叉")
    must_raise_value_error(lambda: verify_disjoint_splits(TRAIN, VALID, TEST + [(0, 1)]), "验证/测试交叉")
    must_raise_value_error(lambda: verify_disjoint_splits([(-1, 0)], VALID, TEST), "非法用户编号")
    # 在同一张训练图上，留出正例不能凭空多出边。
    original = build_bipartite_adjacency(TRAIN, 2, 3).to_dense()
    ensure(original[0, 3].item() == 0, "验证边 (U0,I1) 不应进入训练图")
    ensure(original[1, 4].item() == 0, "测试边 (U1,I2) 不应进入训练图")


def test_f_day7_integration(run_dir: str):
    """F：用 Day 7 的官方 MovieLens 数据产物完整构图并核对图源。"""
    from run_day10 import read_pairs, run_day10
    root = Path(run_dir)
    conf = json.loads((root / "config.json").read_text(encoding="utf-8"))
    ensure(conf.get("dataset") == "ml100k", "必须使用 Day 7 的 MovieLens-100K 目录")
    ensure(conf.get("n_users") == 943 and conf.get("n_items") == 1682, "用户与物品数量错误")
    ensure(sum(int(conf[k]) for k in ("n_train", "n_valid", "n_test")) == 100000,
           "原始 100,000 条交互应全部进入冻结划分")
    train = read_pairs(root / "train_pairs.tsv")
    valid = read_pairs(root / "valid_pairs.tsv")
    test = read_pairs(root / "test_pairs.tsv")
    ensure(len(train) == conf["n_train"], "训练交互数量与配置不一致")
    ensure(len(valid) == conf["n_valid"] and len(test) == conf["n_test"], "留出交互与配置不一致")
    ensure(len(valid) > 0 and len(test) > 0, "需要独立的验证/测试交互")
    with tempfile.TemporaryDirectory() as work:
        output = Path(work)
        summary = run_day10(root, output)
        ensure(summary["graph_source"] == "train_only", "训练图必须只来自 train")
        ensure(summary["num_users"] == 943 and summary["num_items"] == 1682,
               "全局用户与物品数量错误")
        ensure(summary["num_nodes"] == 2625 and summary["adjacency_shape"] == [2625, 2625],
               "图应有 2625 个节点")
        ensure(summary["num_undirected_edges"] == len(train), "训练边数量错误")
        ensure(summary["num_stored_directed_entries"] == 2 * len(train), "双向边数量错误")
        ensure(summary["num_valid_pairs_audited"] == len(valid), "验证交互审计错误")
        ensure(summary["num_test_pairs_audited"] == len(test), "测试交互审计错误")
        for name in ("adjacency.pt", "degrees.pt", "graph_summary.json"):
            ensure((output / name).is_file(), f"缺少图实验文件：{name}")
        a = torch.load(output / "adjacency.pt", map_location="cpu", weights_only=True).coalesce()
        deg = torch.load(output / "degrees.pt", map_location="cpu", weights_only=True)
        ensure(a.layout == torch.sparse_coo and a._nnz() == 2 * len(train), "保存的稀疏图错误")
        ensure(torch.equal(a.to_dense(), a.to_dense().T), "训练无向图不是对称的")
        ensure(int(deg.sum().item()) == 2 * len(train), "训练图总度数不正确")
        # 对留出交互逐条验证：不允许将它们误作为训练边加入。
        dense = a.to_dense()
        ensure(all(dense[u, 943 + i].item() == 0 for u, i in valid + test),
               "发现验证/测试集边泄漏到训练图")
        ensure(all(dense[u, 943 + i].item() == 1 for u, i in train),
               "某些训练交互没有进入图")
        print(f"    MovieLens 图：节点={summary['num_nodes']}，训练无向边={len(train)}，"
              f"存储双向条目={a._nnz()}")


def main():
    parser = argparse.ArgumentParser(description="Day 10：图论、二部图构造自动验收")
    parser.add_argument("--run-dir", default=None, help="Day 7 MovieLens 训练产物目录")
    args = parser.parse_args()
    cases = [
        ("A. 全局节点编号及越界检查", 10, test_a_mapping),
        ("B. 二值用户—物品交互矩阵 R", 15, test_b_interactions),
        ("C. 对称二部图 COO 邻接矩阵 A", 25, test_c_sparse_adjacency),
        ("D. 节点度、邻居与握手定理", 15, test_d_degree_neighbors),
        ("E. 孤立节点、重复交互与防泄漏", 15, test_e_boundaries_leakage),
    ]
    passed = 0
    total = 0
    for title, points, fn in cases:
        try:
            fn()
            passed += 1
            total += points
            print(f"[PASS] {title} (+{points})")
        except Exception as exc:
            print(f"[FAIL] {title}：{type(exc).__name__}：{exc}")
            traceback.print_exc(limit=2)
    if args.run_dir is None:
        print("[SKIP] F. Day 7 MovieLens-100K 数据集成构图 (+20)：未提供 --run-dir")
    else:
        try:
            test_f_day7_integration(args.run_dir)
            passed += 1
            total += 20
            print("[PASS] F. Day 7 MovieLens-100K 数据集成构图 (+20)")
        except Exception as exc:
            print(f"[FAIL] F. 数据集成构图：{type(exc).__name__}：{exc}")
            traceback.print_exc(limit=2)
    print(f"\nDay 10 自动验收：{passed}/6 项通过，{total}/100 分")
    if total < 100:
        print("尚未达到完整验收：还需真实 Day 7 数据集成及理论手算。")
        raise SystemExit(1)
    print("自动验收通过；请继续完成 Day10_notes.md 的手算与问答。")


if __name__ == "__main__":
    main()
