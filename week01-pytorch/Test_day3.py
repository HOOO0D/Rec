"""Day 3 / Week 1 综合验收。

将本文件与 simple_recommender.py 放在同一目录，运行：
    python Test_Day3.py

本文件只提供测试，不包含模型或训练的参考答案。
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 始终从测试脚本所在目录导入，而不是依赖当前工作目录。
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from simple_recommender import SimpleRecommender, train_model, save_model, load_model
except Exception as exc:
    print("[IMPORT FAILED] 请完成 simple_recommender.py 中约定的四个接口。")
    print(f"具体错误：{type(exc).__name__}: {exc}")
    raise SystemExit(1)


# 所有测试统一使用 CPU，避免 GPU/CPU 状态与设备差异影响验收。
DEVICE = torch.device("cpu")
SCORES = [15, 20, 20, 30, 15]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_a_model_and_shapes() -> None:
    """A：模型结构和 Batch 形状，15 分。"""
    model = SimpleRecommender(n_users=4, n_items=5, embedding_dim=3).to(DEVICE)
    check(isinstance(model, nn.Module), "模型必须继承 nn.Module")
    check(isinstance(model.user_embedding, nn.Embedding), "user_embedding 必须是 nn.Embedding")
    check(isinstance(model.item_embedding, nn.Embedding), "item_embedding 必须是 nn.Embedding")
    check(tuple(model.user_embedding.weight.shape) == (4, 3), "用户 Embedding shape 应为 [4,3]")
    check(tuple(model.item_embedding.weight.shape) == (5, 3), "物品 Embedding shape 应为 [5,3]")

    users = torch.tensor([0, 1, 3], dtype=torch.long)
    items = torch.tensor([2, 4, 1], dtype=torch.long)
    scores = model(users, items)
    check(isinstance(scores, torch.Tensor), "forward 必须返回 Tensor")
    check(scores.shape == (3,), f"Batch=3 时应返回 [3]，实际为 {tuple(scores.shape)}")
    check(scores.is_floating_point(), "评分必须是浮点数")
    check(torch.isfinite(scores).all().item(), "评分不能包含 NaN/Inf")

    single = model(users[:1], items[:1])
    check(single.shape == (1,), "Batch=1 时应返回 [1]，不要使用无参数的 squeeze()")


def test_b_hand_computed_dot_product() -> None:
    """B：固定权重的手算点积，20 分。"""
    model = SimpleRecommender(n_users=3, n_items=4, embedding_dim=2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([
            [1., 2.], [3., 4.], [5., 6.]
        ]))
        model.item_embedding.weight.copy_(torch.tensor([
            [2., 1.], [0., 3.], [-1., 2.], [2., -2.]
        ]))

    users = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    items = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    # 1*2+2*1=4；3*0+4*3=12；5*(-1)+6*2=7；3*2+4*(-2)=-2
    expected = torch.tensor([4., 12., 7., -2.])
    actual = model(users, items)
    check(actual.shape == (4,), f"点积输出应为 [4]，实际为 {tuple(actual.shape)}")
    check(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6),
          f"应逐样本点积，不是矩阵乘法或求和混淆。期望 {expected.tolist()}，实际 {actual.tolist()}")


def test_c_gradients_and_one_step() -> None:
    """C：Embedding 梯度与真实参数更新，20 分。"""
    model = SimpleRecommender(n_users=4, n_items=5, embedding_dim=2)
    with torch.no_grad():
        model.user_embedding.weight.fill_(0.5)
        model.item_embedding.weight.fill_(0.25)

    users = torch.tensor([0, 2], dtype=torch.long)
    items = torch.tensor([1, 3], dtype=torch.long)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    old_u = model.user_embedding.weight.detach().clone()
    old_i = model.item_embedding.weight.detach().clone()

    optimizer.zero_grad()
    loss = (model(users, items) - torch.tensor([1., 2.])).square().mean()
    check(loss.ndim == 0, "训练 Loss 应为标量")
    loss.backward()

    grad_u = model.user_embedding.weight.grad
    grad_i = model.item_embedding.weight.grad
    check(grad_u is not None and grad_i is not None, "两个 Embedding 表都必须参与梯度计算")
    check(torch.isfinite(grad_u).all().item() and torch.isfinite(grad_i).all().item(), "梯度出现 NaN/Inf")
    check(grad_u[[0, 2]].abs().sum().item() > 0, "被访问的用户向量需要非零梯度")
    check(grad_i[[1, 3]].abs().sum().item() > 0, "被访问的物品向量需要非零梯度")
    check(torch.allclose(grad_u[[1, 3]], torch.zeros_like(grad_u[[1, 3]])), "未访问的用户行不应产生梯度")
    check(torch.allclose(grad_i[[0, 2, 4]], torch.zeros_like(grad_i[[0, 2, 4]])), "未访问的物品行不应产生梯度")

    optimizer.step()
    check(not torch.allclose(old_u, model.user_embedding.weight.detach()), "用户 Embedding 在 step 后必须变化")
    check(not torch.allclose(old_i, model.item_embedding.weight.detach()), "物品 Embedding 在 step 后必须变化")


def test_d_full_training() -> None:
    """D：固定数据训练、Loss 下降、参数改变，30 分。"""
    torch.manual_seed(20261019)
    true_user = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]])
    true_item = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.], [2., -1.]])
    ratings_matrix = true_user @ true_item.T       # [4,5]，只用于生成可学习的合成评分
    user_ids = torch.arange(4).repeat_interleave(5).long()
    item_ids = torch.arange(5).repeat(4).long()
    ratings = ratings_matrix.flatten().float()
    loader = DataLoader(
        TensorDataset(user_ids, item_ids, ratings),
        batch_size=5,
        shuffle=True,
        generator=torch.Generator().manual_seed(20261019),
    )

    model = SimpleRecommender(n_users=4, n_items=5, embedding_dim=2)
    old_u = model.user_embedding.weight.detach().clone()
    old_i = model.item_embedding.weight.detach().clone()
    model.eval()
    with torch.no_grad():
        initial_mse = torch.mean((model(user_ids, item_ids) - ratings) ** 2).item()

    epochs = 250
    history = train_model(model, loader, epochs=epochs, lr=0.05)
    check(isinstance(history, (list, tuple)), "train_model 需返回每个 epoch 的平均 Loss 列表")
    check(len(history) == epochs, f"需要返回 {epochs} 个 epoch 的 Loss，当前只有 {len(history)} 个")
    check(all(isinstance(x, (int, float)) and math.isfinite(x) for x in history), "Loss 历史必须是有限数值")
    check(history[-1] < history[0], "训练记录中最终 Loss 应小于首轮 Loss")

    model.eval()
    with torch.no_grad():
        final_mse = torch.mean((model(user_ids, item_ids) - ratings) ** 2).item()
    print(f"    初始全数据 MSE = {initial_mse:.6f}，最终全数据 MSE = {final_mse:.6f}")
    check(math.isfinite(final_mse), "最终评分 MSE 不是有限数值")
    check(final_mse < initial_mse * 0.15, "模型没有显著降低 MSE（要求低于初始的 15%）")
    check(final_mse < 0.05, "合成数据应能被低维点积模型较好拟合（最终 MSE < 0.05）")
    check(not torch.allclose(old_u, model.user_embedding.weight.detach()), "训练后用户参数没有更新")
    check(not torch.allclose(old_i, model.item_embedding.weight.detach()), "训练后物品参数没有更新")


def test_e_save_and_load() -> None:
    """E：保存模型权重，重新实例化后预测一致，15 分。"""
    torch.manual_seed(1234)
    model = SimpleRecommender(n_users=4, n_items=5, embedding_dim=3)
    users = torch.tensor([0, 2, 3, 1], dtype=torch.long)
    items = torch.tensor([4, 2, 0, 1], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        before = model(users, items).detach().clone()

    with tempfile.TemporaryDirectory(prefix="day3_check_") as tmp:
        path = Path(tmp) / "model_weights.pth"
        save_model(model, path)
        check(path.is_file(), "save_model 没有创建指定的文件")
        restored = load_model(path, n_users=4, n_items=5, embedding_dim=3)
        check(isinstance(restored, SimpleRecommender), "load_model 应返回新的 SimpleRecommender 实例")
        restored.eval()
        with torch.no_grad():
            after = restored(users, items)
        check(torch.allclose(before, after, atol=1e-7, rtol=1e-7),
              f"保存/加载后预测不一致：原 {before.tolist()}，重载 {after.tolist()}")
        for key, tensor in model.state_dict().items():
            check(key in restored.state_dict(), f"重载模型缺少参数 {key}")
            check(torch.equal(tensor, restored.state_dict()[key]), f"重载参数 {key} 与原权重不同")


def main() -> int:
    tests = [
        ("A. 模型结构与 Batch 形状", test_a_model_and_shapes),
        ("B. 手算点积一致性", test_b_hand_computed_dot_product),
        ("C. 梯度与一步参数更新", test_c_gradients_and_one_step),
        ("D. 完整训练与 Loss 收敛", test_d_full_training),
        ("E. 保存、重载与预测一致", test_e_save_and_load),
    ]
    total = 0
    passed = 0
    print("=" * 60)
    print("Day 3 / Week 1 PyTorch 综合自动验收")
    print("=" * 60)
    for (name, test), points in zip(tests, SCORES):
        try:
            test()
        except Exception as exc:
            print(f"[FAIL] {name} (0/{points})")
            print(f"       {type(exc).__name__}: {exc}")
        else:
            total += points
            passed += 1
            print(f"[PASS] {name} (+{points})")
    print("-" * 60)
    print(f"自动测试：{passed}/5 项通过，得分 {total}/100")
    print("Week 1 通过条件：自动测试 5/5 全 PASS，另完成 week01_notes.md 与口头验收。")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
