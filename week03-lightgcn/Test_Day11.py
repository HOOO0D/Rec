"""Day 11：对称归一化与一层稀疏聚合自动验收，共 100 分。

运行：
    python Test_Day11.py
    python Test_Day11.py --graph-dir experiments/day10_ml100k

A–E 共 80 分，只依赖合成小图；F（20 分）检查 Day 10 图实验产物。
本测试是独立验收，不能代替纸笔推导和理论回答。
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import traceback
from pathlib import Path

import torch

from graph_aggregation import (
    degrees_from_adjacency,
    inverse_sqrt_degrees,
    normalize_bipartite_adjacency,
    manual_node_aggregate,
    sparse_aggregate,
)


def check(ok, reason: str) -> None:
    """检查结果并输出中文错误信息。"""
    if not bool(ok):
        raise AssertionError(reason)


def near(actual, expected, reason: str, atol: float = 1e-5) -> None:
    """使用浮点误差容限核对矩阵和向量。"""
    check(torch.allclose(torch.as_tensor(actual).float(), torch.as_tensor(expected).float(), atol=atol, rtol=1e-5),
          f"{reason}；实际：{actual}，期望：{expected}")


def expect_value_error(action, reason: str) -> None:
    """非法输入应明确抛出 ValueError。"""
    try:
        action()
    except ValueError:
        return
    raise AssertionError(f"{reason}：预期抛 ValueError")


def example_graph() -> torch.Tensor:
    """2 个用户、3 个物品；节点顺序为 U0、U1、I0、I1、I2。"""
    return torch.tensor([
        [0., 0., 1., 0., 1.],
        [0., 0., 1., 1., 0.],
        [1., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
    ]).to_sparse().coalesce()


def initial_embeddings() -> torch.Tensor:
    """与学习笔记手算题使用同一组二维向量。"""
    return torch.tensor([
        [1., 0.],
        [0., 2.],
        [2., 0.],
        [0., 4.],
        [3., 1.],
    ])


def expected_dense_normalized() -> torch.Tensor:
    """预先手算好的稠密归一化结果，仅用于小图测试对照。"""
    c = 1.0 / math.sqrt(2.0)
    return torch.tensor([
        [0., 0., .5, 0., c],
        [0., 0., .5, c, 0.],
        [.5, .5, 0., 0., 0.],
        [0., c, 0., 0., 0.],
        [c, 0., 0., 0., 0.],
    ])


def test_a_degrees():
    """A：节点度、逆平方根及孤立节点数值。"""
    A = example_graph()
    d = degrees_from_adjacency(A)
    check(d.shape == (5,) and d.is_floating_point(), "度向量应该是 [5] 浮点 Tensor")
    near(d, torch.tensor([2., 2., 2., 1., 1.]), "节点度计算错误")
    near(inverse_sqrt_degrees(d),
         torch.tensor([1 / math.sqrt(2)] * 3 + [1., 1.]), "逆平方根度错误")
    v = inverse_sqrt_degrees(torch.tensor([0., 1., 4., 9.]))
    near(v, torch.tensor([0., 1., .5, 1 / 3]), "包含零度数时应正确求逆平方根")
    check(torch.isfinite(v).all(), "孤立节点不能产生 NaN/Inf")
    expect_value_error(lambda: inverse_sqrt_degrees(torch.tensor([-1., 2.])), "负节点度")
    expect_value_error(lambda: inverse_sqrt_degrees(torch.tensor([float('nan')])), "非有限节点度")
    expect_value_error(lambda: degrees_from_adjacency(torch.eye(3)), "输入不是稀疏 COO")


def test_b_normalization():
    """B：每条边的对称归一化权重、对称性、稀疏格式和不增加自环。"""
    A = example_graph()
    before = A.values().clone()
    norm = normalize_bipartite_adjacency(A)
    check(norm.layout == torch.sparse_coo, "归一化必须返回稀疏 COO")
    check(norm.is_coalesced(), "归一化输出必须 coalesce")
    check(norm.dtype == torch.float32, "归一化边权应为 float32")
    check(norm.shape == A.shape, "归一化不能改变图形状")
    check(norm._nnz() == A._nnz(), "不能凭空增加或删除边")
    check(torch.equal(norm.indices(), A.indices()), "归一化只能改变原边权，不能改变坐标")
    near(norm.to_dense(), expected_dense_normalized(), "对称归一化矩阵与手算不一致")
    near(norm.to_dense(), norm.to_dense().T, "无向图的归一化结果应对称")
    near(A.values(), before, "归一化不能原地修改输入图")
    near(torch.diag(norm.to_dense()), torch.zeros(5), "LightGCN 传播图中不加自环")
    near(norm.to_dense().sum(dim=1),
         expected_dense_normalized().sum(dim=1), "归一化后的每行和错误")
    check(not torch.allclose(norm.to_dense().sum(dim=1), torch.ones(5)),
          "对称归一化一般不要求行和为 1；请不要误写成 D^-1 A")


def test_c_manual_aggregation():
    """C：逐边手算消息聚合与预期节点 Embedding。"""
    norm = normalize_bipartite_adjacency(example_graph())
    features = initial_embeddings()
    c = 1 / math.sqrt(2)
    target = torch.tensor([
        [1 + 3 * c, c],
        [1., 4 * c],
        [.5, 1.],
        [0., 2 * c],
        [c, 0.],
    ])
    for u in range(5):
        actual = manual_node_aggregate(norm, features, u)
        check(actual.shape == (2,), f"节点 {u} 的手工聚合结果应为 [d]")
        near(actual, target[u], f"节点 {u} 的手算传播结果不匹配")
    expect_value_error(lambda: manual_node_aggregate(norm, features, 5), "节点编号越界")
    expect_value_error(lambda: manual_node_aggregate(norm, features, -1), "负节点编号")


def test_d_sparse_matmul_gradient():
    """D：sparse.mm 输出、稠密等价性、梯度可回传到 Embedding。"""
    norm = normalize_bipartite_adjacency(example_graph())
    features = initial_embeddings().requires_grad_(True)
    out = sparse_aggregate(norm, features)
    check(out.shape == features.shape and out.layout == torch.strided, "一次传播必须返回 [N,d] 的稠密张量")
    near(out, norm.to_dense() @ features, "稀疏传播结果与稠密矩阵乘法不同")
    check(out.grad_fn is not None, "传播必须保留计算图，不能 detach()")
    out.square().sum().backward()
    check(features.grad is not None and torch.isfinite(features.grad).all(), "梯度未正确回传到初始特征")
    check(float(features.grad.abs().sum()) > 0., "梯度不应全部为零")
    expect_value_error(lambda: sparse_aggregate(norm, torch.zeros(4, 2)), "特征节点数不匹配")
    expect_value_error(lambda: sparse_aggregate(norm, torch.zeros(5)), "特征应为二维 [N,d]")


def test_e_isolated_and_empty():
    """E：孤立节点、全空图、极小图，不能出现非有限值。"""
    A = torch.zeros((4, 4), dtype=torch.float32)
    A[0, 1] = A[1, 0] = 1.0
    sparse = A.to_sparse().coalesce()
    degree = degrees_from_adjacency(sparse)
    near(degree, torch.tensor([1., 1., 0., 0.]), "有两个孤立节点时的度数不正确")
    norm = normalize_bipartite_adjacency(sparse)
    check(torch.isfinite(norm.values()).all(), "归一化后的权重不能出现 NaN/Inf")
    features = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    out = sparse_aggregate(norm, features)
    near(out, torch.tensor([[3., 4.], [1., 2.], [0., 0.], [0., 0.]]),
         "孤立节点一跳聚合应得到全零向量")
    near(manual_node_aggregate(norm, features, 2), torch.zeros(2), "孤立节点手动聚合应为零")
    no_edges = torch.zeros((3, 3), dtype=torch.float32).to_sparse().coalesce()
    no_norm = normalize_bipartite_adjacency(no_edges)
    check(no_norm._nnz() == 0 and no_norm.shape == (3, 3), "空图仍应保留三节点的形状")
    near(sparse_aggregate(no_norm, torch.ones(3, 4)), torch.zeros(3, 4), "全空图聚合应全部为零")


def test_f_day10_integration(graph_dir: str):
    """F：读取 Day 10 图，核对数据来源，并真正保存归一化与一次传播结果。"""
    root = Path(graph_dir)
    info = json.loads((root / "graph_summary.json").read_text(encoding="utf-8"))
    check(info.get("num_users") == 943 and info.get("num_items") == 1682,
          "集成验收需要 Day 10 的 MovieLens-100K 图；不能用 2×3 玩具图代替")
    check(info.get("graph_source") == "train_only", "只能使用训练交互构图")
    check(info.get("has_self_loops") is False and info.get("has_normalization") is False,
          "Day 10 输入应该是无自环、未归一化二部图")
    check(0 < info.get("num_train_pairs", 0) < 100000, "训练边数应是 MovieLens-100K 留出验证/测试后的数据量")
    check(info.get("num_valid_pairs_audited", 0) > 0 and info.get("num_test_pairs_audited", 0) > 0,
          "应记录验证/测试集审计数量")
    from run_day11 import run_day11
    with tempfile.TemporaryDirectory() as temp:
        report = run_day11(root, temp)
        output = Path(temp)
        for name in ("normalized_adjacency.pt", "inverse_sqrt_degrees.pt", "one_hop_sample.pt", "normalization_summary.json"):
            check((output / name).is_file(), f"缺少 Day 11 实验产物：{name}")
        norm = torch.load(output / "normalized_adjacency.pt", map_location="cpu", weights_only=True).coalesce()
        inv = torch.load(output / "inverse_sqrt_degrees.pt", map_location="cpu", weights_only=True)
        sample = torch.load(output / "one_hop_sample.pt", map_location="cpu", weights_only=True)
        original = torch.load(root / "adjacency.pt", map_location="cpu", weights_only=True).coalesce()
        saved_deg = torch.load(root / "degrees.pt", map_location="cpu", weights_only=True).float()
        check(tuple(norm.shape) == (2625, 2625), "MovieLens-100K 图节点数应该是 2625")
        check(norm._nnz() == 2 * info["num_train_pairs"], "归一化过程中不能增删边")
        check(torch.equal(norm.indices(), original.indices()), "归一化后所有边的坐标必须不变")
        near(inv, inverse_sqrt_degrees(saved_deg), "保存的逆平方根度数错误")
        row, col = norm.indices()
        expected_values = inv[row] * original.values() * inv[col]
        near(norm.values(), expected_values, "真实图的边权未按两端度数对称归一化")
        check(torch.isfinite(norm.values()).all(), "真实图出现非法边权")
        check(report["graph_source"] == "train_only" and report["normalization"] == "symmetric_degree_no_self_loops",
              "摘要中的训练图来源和归一化方式错误")
        check(report.get("train_pairs_sha256") == info.get("train_pairs_sha256"),
              "训练交互指纹不一致，不能静默更换图来源")
        check(sample["initial"].shape == (5, 3) and sample["one_hop"].shape == (5, 3),
              "单次传播检查样例的形状应该是 [5,3]")
        check(torch.isfinite(sample["one_hop"]).all(), "单次传播样例出现 NaN/Inf")
        print(f"    MovieLens-100K：节点={info['num_nodes']}，训练无向边={info['num_train_pairs']}，"
              f"COO 非零项={norm._nnz()}，孤立节点={report['num_isolated_nodes']}")


def main():
    parser = argparse.ArgumentParser(description="Day 11：LightGCN 归一化图与邻域传播测试")
    parser.add_argument("--graph-dir", default=None, help="Day 10 输出的 MovieLens 图目录")
    args = parser.parse_args()
    tests = [
        ("A. 节点度与安全逆平方根", 15, test_a_degrees),
        ("B. 对称归一化边权与稀疏格式", 25, test_b_normalization),
        ("C. 逐节点手算消息聚合", 15, test_c_manual_aggregation),
        ("D. torch.sparse.mm 与梯度验证", 15, test_d_sparse_matmul_gradient),
        ("E. 孤立节点和空图边界", 10, test_e_isolated_and_empty),
    ]
    score = passed = 0
    for label, weight, func in tests:
        try:
            func()
        except Exception as exc:
            print(f"[FAIL] {label}：{type(exc).__name__}：{exc}")
            traceback.print_exc(limit=2)
        else:
            score += weight
            passed += 1
            print(f"[PASS] {label} (+{weight})")
    if args.graph_dir is None:
        print("[SKIP] F. Day 10 MovieLens 图归一化与保存 (+20)：未指定 --graph-dir")
    else:
        try:
            test_f_day10_integration(args.graph_dir)
        except Exception as exc:
            print(f"[FAIL] F. Day 10 MovieLens 图归一化与保存：{type(exc).__name__}：{exc}")
            traceback.print_exc(limit=2)
        else:
            score += 20
            passed += 1
            print("[PASS] F. Day 10 MovieLens 图归一化与保存 (+20)")
    print(f"\nDay 11 自动验收：{passed}/6 项，{score}/100 分")
    if score < 100:
        print("全部完成还需要真实 Day 10 MovieLens 图，并完成理论笔记。")
        raise SystemExit(1)
    print("自动测试全部通过，还需要独立完成公式推导与解释。")


if __name__ == "__main__":
    main()
