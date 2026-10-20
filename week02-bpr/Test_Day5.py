"""Day 5 Matrix Factorization 自动验收（100 分）

执行：python Test_Day5.py
检查同目录的 model.py（或 model_Day5.py）是否实现规定接口。
测试不提供实现答案，且只在 CPU 上使用合成数据，不下载 MovieLens。
"""

import importlib
import math
import traceback
import torch

try:
    mf = importlib.import_module("model")
except ModuleNotFoundError as ex:
    if ex.name != "model":
        raise
    mf = importlib.import_module("model_Day5")


def require(ok, msg):
    if not ok:
        raise AssertionError(msg)


def allclose(a, b, atol=1e-6):
    require(torch.allclose(a, b, atol=atol, rtol=1e-6),
            f"数值不一致\n实际: {a}\n期望: {b}")


def assign_known_weights(model):
    p = torch.tensor([[1., 2.], [3., -1.], [2., 2.]])
    q = torch.tensor([[1., 0.], [0., 2.], [2., -1.], [-1., 1.]])
    with torch.no_grad():
        model.user_embedding.weight.copy_(p)
        model.item_embedding.weight.copy_(q)
    return p, q


def test_a():
    model = mf.MatrixFactorization(3, 4, 2)
    require(isinstance(model, torch.nn.Module), "MF 必须继承 nn.Module")
    require(isinstance(model.user_embedding, torch.nn.Embedding), "缺少 user_embedding")
    require(isinstance(model.item_embedding, torch.nn.Embedding), "缺少 item_embedding")
    require(tuple(model.user_embedding.weight.shape) == (3, 2), "用户表形状错误")
    require(tuple(model.item_embedding.weight.shape) == (4, 2), "物品表形状错误")
    require(model.user_embedding.weight.requires_grad, "用户表应可训练")
    require(model.item_embedding.weight.requires_grad, "物品表应可训练")
    for size in (1, 4, 7):
        users = torch.arange(size, dtype=torch.long) % 3
        items = torch.arange(size, dtype=torch.long) % 4
        out = model(users, items)
        require(tuple(out.shape) == (size,), f"Batch={size} 时应输出 [{size}]，实际 {tuple(out.shape)}")
        require(out.dtype.is_floating_point, "得分应是浮点张量")
    return "Embedding 形状及 B=1/4/7 成对预测正确"


def test_b():
    model = mf.MatrixFactorization(3, 4, 2)
    assign_known_weights(model)
    users = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    items = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    actual = model(users, items)
    expected = torch.tensor([1., -2., 7., 0.])
    allclose(actual, expected)
    allclose(model(torch.tensor([2]), torch.tensor([2])), torch.tensor([2.]))
    return "与手算点积 [1,-2,7,0] 一致"


def test_c():
    model = mf.MatrixFactorization(3, 4, 2)
    p, q = assign_known_weights(model)
    scores = model.score_all()
    expected = p @ q.T
    require(tuple(scores.shape) == (3, 4), "score_all 输出必须是 [n_users,n_items]")
    allclose(scores, expected)
    require(int(torch.linalg.matrix_rank(scores)) <= 2, "预测矩阵秩应 <= embedding_dim")
    ids_u = torch.tensor([0, 1, 2], dtype=torch.long)
    ids_i = torch.tensor([3, 0, 1], dtype=torch.long)
    allclose(model(ids_u, ids_i), scores[ids_u, ids_i])
    return "全量评分 P@Q.T 与成对评分一致，秩 <= 2"


def test_d():
    model = mf.MatrixFactorization(3, 4, 2)
    p, q = assign_known_weights(model)
    penalty = model.l2_regularization()
    require(isinstance(penalty, torch.Tensor) and penalty.ndim == 0, "正则项必须为 scalar Tensor")
    expected = (p ** 2).sum() + (q ** 2).sum()
    allclose(penalty, expected)
    model.zero_grad()
    penalty.backward()
    allclose(model.user_embedding.weight.grad, 2 * p)
    allclose(model.item_embedding.weight.grad, 2 * q)
    return f"L2 平方和={expected.item():g}，梯度分别为 2P/2Q"


def test_e():
    model = mf.MatrixFactorization(3, 4, 2)
    p, q = assign_known_weights(model)
    users = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    items = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    ratings = torch.tensor([2., 1., 1., -2.])
    observed_predictions = (p[users] * q[items]).sum(dim=1)
    expected_mse = ((observed_predictions - ratings) ** 2).mean()
    for lam in (0., 0.01):
        got = mf.mf_loss(model, users, items, ratings, reg_lambda=lam)
        require(got.ndim == 0, "mf_loss 返回的必须是 scalar Tensor")
        allclose(got, expected_mse + lam * ((p*p).sum() + (q*q).sum()))

    torch.manual_seed(20261019)
    fit = mf.MatrixFactorization(3, 4, 2)
    true_p = torch.tensor([[1.0, 0.2], [0.4, -0.8], [-0.5, 1.0]])
    true_q = torch.tensor([[0.8, 0.1], [-0.4, 0.8], [0.7, -0.2], [0.1, -1.0]])
    train_users = torch.arange(3).repeat_interleave(4)
    train_items = torch.arange(4).repeat(3)
    targets = (true_p[train_users] * true_q[train_items]).sum(dim=1)
    with torch.no_grad():
        initial_mse = ((fit(train_users, train_items) - targets) ** 2).mean().item()
    history = mf.train_mf(
        fit, train_users, train_items, targets,
        epochs=300, lr=0.05, reg_lambda=0.0,
    )
    require(len(history) == 300, "train_mf 返回的 Loss 历史长度应为 epochs")
    require(all(isinstance(x, (float, int)) and math.isfinite(float(x)) for x in history),
            "Loss 历史必须由有限数值构成")
    with torch.no_grad():
        final_mse = ((fit(train_users, train_items) - targets) ** 2).mean().item()
    require(final_mse < 0.02 and final_mse < 0.15 * initial_mse,
            f"训练没有收敛：初始 MSE={initial_mse:.5f} 最终 MSE={final_mse:.5f}")
    require(history[-1] < history[0], "Loss 历史没有下降")
    return f"观测样本 MSE+L2 计算正确；训练 {initial_mse:.5f} -> {final_mse:.5f}"


def main():
    print("="*60)
    print("Day 5｜矩阵分解 MF 自动验收")
    print("="*60)
    cases = [
        ("A. Embedding 和 Batch 形状", 15, test_a),
        ("B. 手算向量点积", 20, test_b),
        ("C. 全量评分矩阵与低秩", 20, test_c),
        ("D. L2 正则化与梯度", 20, test_d),
        ("E. MF Loss 与训练闭环", 25, test_e),
    ]
    passed = score = 0
    for title, weight, func in cases:
        try:
            description = func()
            passed += 1
            score += weight
            print(f"[PASS] {title} (+{weight})：{description}")
        except Exception as exc:
            print(f"[FAIL] {title}：{type(exc).__name__}: {exc}")
    print("="*60)
    print(f"自动验收：{passed}/5 项通过，{score}/100 分")
    print("达标标准：5/5 PASS + 100/100 + 独立理论验收")
    print("="*60)
    if passed != 5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
