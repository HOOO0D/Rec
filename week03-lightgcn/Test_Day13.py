"""Day 13｜LightGCN 最小模型前向与梯度自动验收：满分 100。

命令：
  python Test_Day13.py
  python Test_Day13.py --graph-dir experiments/day11_ml100k

A–E 共 80 分（合成小图）；F 20 分（接入本地 Day 11 的训练图）。
注意：此测试不包含 Day 14 的 BPR 训练，也不检查 Recall/NDCG 指标。
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import traceback
from pathlib import Path
import torch
from lightgcn_model import LightGCN
from run_day13 import run_day13


def assert_true(condition, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, message: str, *, atol: float = 1e-5):
    if not torch.allclose(actual, expected, atol=atol, rtol=1e-5):
        raise AssertionError(f"{message}\n实际={actual}\n预期={expected}")


def assert_raises_value_error(callback, message: str):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError(message + "：应抛 ValueError")


def small_adjacency() -> torch.Tensor:
    """来自 Day 10/11 的 2 用户 + 3 物品图；无自环，已对称归一化。"""
    c = 1.0 / math.sqrt(2)
    return torch.tensor([
        [0., 0., .5, 0., c],
        [0., 0., .5, c, 0.],
        [.5, .5, 0., 0., 0.],
        [0., c, 0., 0., 0.],
        [c, 0., 0., 0., 0.],
    ], dtype=torch.float32).to_sparse().coalesce()


def initial_embeddings() -> torch.Tensor:
    return torch.tensor([
        [1., 0.], [0., 2.], [2., 0.], [0., 4.], [3., 1.],
    ], dtype=torch.float32)


def make_model(n_layers: int = 2, *, weights=None, graph=None) -> LightGCN:
    if graph is None:
        graph = small_adjacency()
    model = LightGCN(n_users=2, n_items=3, embedding_dim=2,
                     n_layers=n_layers, normalized_adjacency=graph,
                     layer_weights=weights)
    with torch.no_grad():
        e0 = initial_embeddings()
        model.user_embedding.weight.copy_(e0[:2])
        model.item_embedding.weight.copy_(e0[2:])
    return model


def expected_layer_1() -> torch.Tensor:
    c = 1 / math.sqrt(2)
    return torch.tensor([
        [1+3*c, c],
        [1, 4*c],
        [.5, 1.],
        [0, 2*c],
        [c, 0],
    ], dtype=torch.float32)


def expected_layer_2() -> torch.Tensor:
    c = 1 / math.sqrt(2)
    return torch.tensor([
        [.75, .5],
        [.25, 1.5],
        [1 + 1.5*c, 2.5*c],
        [c, 2.],
        [1.5 + c, .5],
    ], dtype=torch.float32)


def test_a_initialization():
    """A：初始化、参数数量、图缓冲区及不合法参数（10 分）。"""
    torch.manual_seed(123)
    model = LightGCN(2, 3, 4, 2, small_adjacency())
    assert_true(model.user_embedding.weight.shape == (2, 4), "用户表形状应为 [U,d]")
    assert_true(model.item_embedding.weight.shape == (3, 4), "物品表形状应为 [I,d]")
    assert_true(sum(p.numel() for p in model.parameters() if p.requires_grad) == 20,
                "标准 LightGCN 不应增加 W/bias 等额外可训练参数")
    buffers = dict(model.named_buffers())
    assert_true("normalized_adjacency" in buffers, "归一化邻接矩阵应注册为 buffer")
    assert_true("layer_weights" in buffers, "层权重应注册为 buffer，不是优化参数")
    assert_true(model.normalized_adjacency.layout == torch.sparse_coo, "应保持稀疏 COO")
    assert_true(model.normalized_adjacency.is_coalesced(), "应合并重复 COO 坐标")
    assert_raises_value_error(lambda: LightGCN(2, 3, 4, -1, small_adjacency()), "负传播层数")
    assert_raises_value_error(lambda: LightGCN(2, 3, 0, 2, small_adjacency()), "Embedding 维数为 0")
    assert_raises_value_error(lambda: LightGCN(2, 3, 4, 2, torch.eye(5)), "不应接受稠密图")
    assert_raises_value_error(lambda: LightGCN(2, 3, 4, 2, torch.eye(4).to_sparse()), "图节点数量不匹配")


def test_b_zero_layer_mf():
    """B：K=0 必须严格退化为 MF（15 分）。"""
    model = make_model(n_layers=0)
    layers = model.propagate_layers()
    assert_true(len(layers) == 1, "K=0 时应只有第 0 层")
    assert_close(layers[0], initial_embeddings(), "第 0 层表示不正确")
    final_users, final_items = model()
    assert_close(final_users, initial_embeddings()[:2], "K=0 用户表示不应改变")
    assert_close(final_items, initial_embeddings()[2:], "K=0 物品表示不应改变")
    u = torch.tensor([0, 1, 0], dtype=torch.long)
    i = torch.tensor([0, 0, 2], dtype=torch.long)
    expected = torch.tensor([2., 0., 3.], dtype=torch.float32)
    assert_close(model.score(u, i), expected, "K=0 配对评分应等于 MF 点积")
    assert_close(model.full_sort_scores(torch.tensor([0, 1])),
                 final_users @ final_items.T, "K=0 全物品预测应为 MF 矩阵乘法")


def test_c_propagation_math():
    """C：K=2 的两层传播、均匀融合、维度（25 分）。"""
    model = make_model(n_layers=2)
    layers = model.propagate_layers()
    assert_true(isinstance(layers, (tuple, list)) and len(layers) == 3,
                "K=2 需要 E0、E1、E2 共三层")
    for k, layer in enumerate(layers):
        assert_true(layer.shape == (5, 2), f"第 {k} 层形状应为 [U+I,d]")
        assert_true(layer.layout == torch.strided, "Embedding 应是稠密张量")
    assert_close(layers[0], initial_embeddings(), "E0 必须拼接用户和物品原始表示")
    assert_close(layers[1], expected_layer_1(), "E1 与 Day 11 手算结果不一致")
    assert_close(layers[2], expected_layer_2(), "E2 与二跳传播手算结果不一致")
    expected_final = (initial_embeddings() + expected_layer_1() + expected_layer_2()) / 3
    combined = model.combine_layers(layers)
    assert_close(combined, expected_final, "均匀层组合 E_final 应为三层平均")
    u, i = model()
    assert_true(u.shape == (2, 2) and i.shape == (3, 2), "用户/物品最终形状不正确")
    assert_close(torch.cat([u, i], dim=0), expected_final, "forward() 必须先融合，再按 U/I 拆开")


def test_d_weights_and_scores():
    """D：非均匀权重、配对评分、全物品评分及边界（15 分）。"""
    model = make_model(2, weights=[0., 0., 1.])
    assert_close(model.combine_layers(model.propagate_layers()), expected_layer_2(),
                 "指定权重 [0,0,1] 应只选择 E2")
    user_emb, item_emb = model()
    uids = torch.tensor([0, 1, 0], dtype=torch.long)
    iids = torch.tensor([0, 1, 2], dtype=torch.long)
    scores = model.score(uids, iids)
    assert_true(scores.shape == (3,), "配对评分应为 [B]，不是 [B,B]")
    assert_close(scores, (user_emb[uids] * item_emb[iids]).sum(dim=1), "配对得分不正确")
    all_scores = model.full_sort_scores(torch.tensor([0, 1], dtype=torch.long))
    assert_true(all_scores.shape == (2, 3), "全物品评分应为 [B,I]")
    assert_close(all_scores, user_emb @ item_emb.T, "全物品评分应为用户与物品表示的矩阵乘法")
    assert_raises_value_error(lambda: make_model(2, weights=[.5, .5]), "层权重数量不正确")
    assert_raises_value_error(lambda: make_model(2, weights=[.2, .2, .2]), "层权重之和不为 1")
    assert_raises_value_error(lambda: make_model(2, weights=[-1., 1., 1.]), "层权重包含负数")
    assert_raises_value_error(lambda: model.score(torch.tensor([2]), torch.tensor([0])), "用户 ID 越界")
    assert_raises_value_error(lambda: model.score(torch.tensor([0]), torch.tensor([3])), "物品 ID 越界")
    assert_raises_value_error(lambda: model.score(torch.tensor([0,1]), torch.tensor([0])), "配对 ID 长度不等")


def test_e_gradient_isolation():
    """E：自动求导、无额外训练参数、孤立节点稳定性（15 分）。"""
    model = make_model(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before_u = model.user_embedding.weight.detach().clone()
    before_i = model.item_embedding.weight.detach().clone()
    optimizer.zero_grad()
    values = model.score(torch.tensor([0, 1], dtype=torch.long),
                         torch.tensor([0, 1], dtype=torch.long))
    loss = values.square().sum()
    assert_true(loss.grad_fn is not None, "LightGCN 前向输出必须可导")
    loss.backward()
    for emb, name in [(model.user_embedding, "用户"), (model.item_embedding, "物品")]:
        grad = emb.weight.grad
        assert_true(grad is not None and torch.isfinite(grad).all(), f"{name} Embedding 梯度不正确")
        assert_true(grad.abs().sum().item() > 0, f"{name} Embedding 梯度不可全为零")
    optimizer.step()
    assert_true(not torch.equal(model.user_embedding.weight.detach(), before_u), "用户权重未更新")
    assert_true(not torch.equal(model.item_embedding.weight.detach(), before_i), "物品权重未更新")

    # 无边图构造孤立节点；单层传播为 0，K=1 时最终向量应是 E0 的一半。
    empty_graph = torch.sparse_coo_tensor(torch.empty((2,0),dtype=torch.long),
                                          torch.empty((0,),dtype=torch.float32), (5,5)).coalesce()
    isolated = make_model(1, graph=empty_graph)
    layers = isolated.propagate_layers()
    assert_close(layers[1], torch.zeros_like(layers[1]), "孤立节点传播一层应为零")
    assert_close(isolated.combine_layers(layers), initial_embeddings() / 2,
                 "孤立节点仍应在最后层组合保留 E0 的贡献")
    uu, ii = isolated()
    assert_true(torch.isfinite(uu).all() and torch.isfinite(ii).all(), "无边图结果不应有 NaN/Inf")


def test_f_integration(graph_dir: str):
    """F：真实 Day 11 训练图集成与文件产物（20 分）。"""
    graph = Path(graph_dir)
    if not graph.is_dir():
        raise FileNotFoundError(f"未找到 Day 11 图目录：{graph}")
    with tempfile.TemporaryDirectory(prefix="day13_test_") as tmp:
        report = run_day13(graph, tmp, embedding_dim=4, n_layers=2, seed=20261031)
        assert_true(report["graph_source"] == "train_only", "图必须来自冻结训练交互")
        assert_true(report["normalization"] == "symmetric_degree_no_self_loops", "必须使用 Day 11 的归一化方案")
        assert_true(report["n_trainable_parameters"] == 4 * report["num_nodes"], "可训练参数数量错误")
        assert_true(report["user_embedding_shape"] == [report["num_users"], 4], "用户特征形状错误")
        assert_true(report["item_embedding_shape"] == [report["num_items"], 4], "物品特征形状错误")
        assert_true(report["is_trained"] is False, "Day 13 不能将随机初始化前向误认为已训练")
        summary_path = Path(tmp) / "forward_summary.json"
        sample_path = Path(tmp) / "forward_sample.pt"
        assert_true(summary_path.is_file() and sample_path.is_file(), "缺少前向检查文件")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert_true(summary == report, "保存的实验摘要与程序返回值不一致")
        sample = torch.load(sample_path, weights_only=True)
        assert_close(sample["paired_scores"], sample["full_sort_scores"][:,0],
                     "配对评分与全物品分数不相符")
        # 固定随机种子，重复执行得到完全一致的输出。
        with tempfile.TemporaryDirectory(prefix="day13_repeat_") as tmp2:
            again = run_day13(graph, tmp2, embedding_dim=4, n_layers=2, seed=20261031)
            sample2 = torch.load(Path(tmp2) / "forward_sample.pt", weights_only=True)
            assert_true(again["n_trainable_parameters"] == report["n_trainable_parameters"],
                        "重复运行的参数数量不一致")
            assert_close(sample2["full_sort_scores"], sample["full_sort_scores"],
                         "相同图、种子与配置下的前向结果应该可复现")


def main():
    parser = argparse.ArgumentParser(description="Day 13：LightGCN 前向实现自动验收")
    parser.add_argument("--graph-dir", default=None, help="Day 11 的真实归一化图目录，用于 F 集成测试")
    args = parser.parse_args()
    cases = [
        ("A. 初始化与参数约束", 10, test_a_initialization),
        ("B. 零层退化 MF", 15, test_b_zero_layer_mf),
        ("C. 两层传播与融合", 25, test_c_propagation_math),
        ("D. 权重和评分接口", 15, test_d_weights_and_scores),
        ("E. 梯度与孤立节点", 15, test_e_gradient_isolation),
    ]
    earned = 0
    for label, points, fn in cases:
        try:
            fn()
        except Exception:
            print(f"[FAIL] {label}：0/{points}")
            traceback.print_exc(limit=2)
        else:
            earned += points
            print(f"[PASS] {label}：+{points}")
    if args.graph_dir:
        try:
            test_f_integration(args.graph_dir)
        except Exception:
            print("[FAIL] F. Day 11 图集成：0/20")
            traceback.print_exc(limit=2)
        else:
            earned += 20
            print("[PASS] F. Day 11 图集成：+20")
    else:
        print("[SKIP] F. 未传入 --graph-dir：不计 20 分")
    print(f"Day 13 验收分数：{earned}/100")
    print("提示：需要完整完成 A–F，并提交手算与实现说明，才视为真正通过 Day 13。")


if __name__ == "__main__":
    main()
