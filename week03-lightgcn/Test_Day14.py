"""Day 14｜LightGCN 训练与 Full Ranking 验收（100 分）。

用法：
  python Test_Day14.py
  python Test_Day14.py --run-dir ../week02-bpr/runs/<你的Day7目录> \\
    --graph-dir experiments/day11_ml100k

A–E：小图单元测试共 80 分；F：真实 MovieLens-100K 一轮集成验收共 20 分。
本脚本不会包含 LightGCN/BPR 的参考解法，只核对接口、公式、梯度与输出结果。
"""
from __future__ import annotations

import argparse
import math
import random
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from lightgcn_model import LightGCN
from lightgcn_train import (
    build_seen_sets, sample_negative_items, calculate_bpr_loss,
    train_one_epoch, evaluate_full_ranking,
)
from run_day14 import run_day14, load_and_audit


def check(cond, msg):
    if not bool(cond):
        raise AssertionError(msg)


def close(actual, expected, msg, atol=1e-5):
    if not torch.allclose(torch.as_tensor(actual), torch.as_tensor(expected), atol=atol, rtol=1e-5):
        raise AssertionError(f"{msg}：实际={actual}，预期={expected}")


def raises(callable_, exc_type=ValueError):
    try:
        callable_()
    except exc_type:
        return
    raise AssertionError(f"应当抛出 {exc_type.__name__}")


def make_sparse_graph(train_pairs: list[tuple[int, int]], n_users: int, n_items: int) -> torch.Tensor:
    """测试专用：只根据正交互生成对称归一化图，不依赖 Day 10/11 代码。"""
    n_nodes = n_users + n_items
    row = [u for u, i in train_pairs] + [i + n_users for u, i in train_pairs]
    col = [i + n_users for u, i in train_pairs] + [u for u, i in train_pairs]
    rows = torch.tensor(row, dtype=torch.long)
    cols = torch.tensor(col, dtype=torch.long)
    deg = torch.bincount(rows, minlength=n_nodes).float()
    vals = (deg[rows] * deg[cols]).rsqrt()
    adj = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n_nodes, n_nodes))
    return adj.coalesce()


def toy_model(train_pairs, n_users, n_items, *, n_layers=2, dim=4):
    return LightGCN(n_users, n_items, dim, n_layers,
                    make_sparse_graph(train_pairs, n_users, n_items))


def test_A_sampling():
    train = [(0, 0), (0, 1), (1, 2), (2, 0)]
    seen = build_seen_sets(train, 3, 5)
    check(seen == [{0, 1}, {2}, {0}], "按用户记录的已见正例不正确")
    raises(lambda: build_seen_sets(train + [(0, 0)], 3, 5))
    raises(lambda: build_seen_sets(train + [(3, 1)], 3, 5))
    users = torch.tensor([0] * 1000 + [1] * 1000 + [2] * 1000, dtype=torch.long)
    negatives = sample_negative_items(users, seen, 5, random.Random(19))
    check(negatives.shape == users.shape and negatives.dtype == torch.long, "负采样输出维度/类型错误")
    check(negatives.device.type == "cpu", "负采样默认返回 CPU 上的 ID 张量")
    for u, j in zip(users.tolist(), negatives.tolist()):
        check(0 <= j < 5 and j not in seen[u], "负采样命中了已见训练正例")
    seen_full = [{0, 1, 2}]
    raises(lambda: sample_negative_items(torch.tensor([0]), seen_full, 3, random.Random(0)))
    # 统一负物品抽样不要求候选数完全相等，但必须能观察到多个可行候选。
    check(set(negatives[:1000].tolist()) == {2, 3, 4}, "合法候选没有均被抽取到")


def test_B_loss():
    train = [(0, 0), (1, 1)]
    torch.manual_seed(3)
    model = toy_model(train, 2, 3, n_layers=0, dim=3)
    u = torch.tensor([0, 1], dtype=torch.long)
    pos = torch.tensor([0, 1], dtype=torch.long)
    neg = torch.tensor([2, 0], dtype=torch.long)
    loss, rank, reg, margin = calculate_bpr_loss(model, u, pos, neg, 0.03)
    user_emb, item_emb = model()
    exp_margin = ((user_emb[u] * item_emb[pos]).sum(1)
                  - (user_emb[u] * item_emb[neg]).sum(1))
    exp_rank = F.softplus(-exp_margin).mean()
    user0 = model.user_embedding(u)
    pos0 = model.item_embedding(pos)
    neg0 = model.item_embedding(neg)
    exp_reg = (user0.pow(2).sum() + pos0.pow(2).sum() + neg0.pow(2).sum()) / (2 * u.numel())
    close(margin, exp_margin, "评分差计算不正确")
    close(rank, exp_rank, "排序损失与 Softplus 公式不一致")
    close(reg, exp_reg, "第 0 层 Batch L2 归一化不正确")
    close(loss, exp_rank + 0.03 * exp_reg, "总损失与正则项系数不一致")
    loss.backward()
    for param in (model.user_embedding.weight, model.item_embedding.weight):
        check(param.grad is not None and torch.isfinite(param.grad).all(), "损失无法回传到初始 Embedding")
    raises(lambda: calculate_bpr_loss(model, u, pos, neg, -0.01))


def test_C_training():
    # 能被 LightGCN 拟合的小图训练问题：不要求每次随机初始化都能达到 100% 排序准确率。
    train = [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4)]
    torch.manual_seed(23)
    model = toy_model(train, 3, 5, n_layers=2, dim=8)
    seen = build_seen_sets(train, 3, 5)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    rng = random.Random(5)
    original = model.user_embedding.weight.detach().clone()
    before = None
    for ep in range(30):
        result = train_one_epoch(model, train, seen, opt, batch_size=5,
                                 reg_lambda=1e-5, rng=rng)
        for name in ('ranking_loss', 'regularization_loss', 'total_loss', 'pair_accuracy'):
            check(name in result and math.isfinite(float(result[name])), f"训练返回值 {name} 缺失或无效")
        check(int(result.get('n_train', -1)) == len(train), '训练样本数有误')
        if ep == 0:
            before = result['ranking_loss']
    check(float(result['ranking_loss']) < min(float(before), 0.2), "训练 30 轮后排序 Loss 未达到小图过拟合要求")
    check(float(result['pair_accuracy']) >= 0.8, "小图排序准确率低于 80%，未实现基本过拟合")
    check(not torch.allclose(original, model.user_embedding.weight), "optimizer 未更新参数")
    for p in model.parameters():
        check(torch.isfinite(p).all(), "训练后模型出现 NaN/Inf")
    check(model.normalized_adjacency.layout == torch.sparse_coo, "不能把稀疏训练图改为稠密矩阵")


def test_D_full_ranking():
    train = [(0, 0), (1, 1)]
    valid = [(0, 2), (0, 3), (1, 4)]
    model = toy_model(train, 2, 5, n_layers=0, dim=2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1., 0.], [0., 1.]]))
        model.item_embedding.weight.copy_(torch.tensor([
            [30., 0.], [0., 30.], [10., 0.], [9., 0.], [0., 10.]
        ]))
    best = evaluate_full_ranking(model, train, valid, k=2, eval_user_batch_size=1)
    check(int(best.get('n_eval_users', -1)) == 2, "验证用户数量不对")
    close(best.get('recall@2', -1), 1., "验证 Recall@2 不正确")
    close(best.get('hit@2', -1), 1., "验证 Hit@2 不正确")
    close(best.get('ndcg@2', -1), 1., "验证 NDCG@2 不正确")
    first = evaluate_full_ranking(model, train, valid, k=1, eval_user_batch_size=2)
    close(first.get('recall@1', -1), .75, "Recall 应按用户先计算再取宏平均")
    close(first.get('hit@1', -1), 1., "Hit@1 计算错误")
    close(first.get('ndcg@1', -1), 1., "NDCG@1 计算错误")
    raises(lambda: evaluate_full_ranking(model, train, [(0,0)], k=2))


def test_E_boundaries():
    model = toy_model([(0, 0)], 2, 3, n_layers=2, dim=3)
    # 无训练边的节点不应产生 NaN；其初始 Embedding 仍属于可训练参数。
    u, i = model()
    check(u.shape == (2, 3) and i.shape == (3, 3), "用户/物品向量形状不正确")
    check(torch.isfinite(u).all() and torch.isfinite(i).all(), "孤立节点导致非有限值")
    with tempfile.TemporaryDirectory() as temp:
        ckpt = Path(temp) / "model.pt"
        torch.save(model.state_dict(), ckpt)
        loaded = toy_model([(0, 0)], 2, 3, n_layers=2, dim=3)
        loaded.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        compare_u, compare_i = loaded()
        close(compare_u, u, "保存重载后用户向量不一致")
        close(compare_i, i, "保存重载后物品向量不一致")
    # 测试只有训练用户可以无负例时必须拒绝训练，防止死循环。
    all_seen = [(0,0), (0,1)]
    seen = build_seen_sets(all_seen, 1, 2)
    raises(lambda: sample_negative_items(torch.tensor([0]), seen, 2, random.Random(0)))


def test_F_movie_lens(run_dir: Path, graph_dir: Path):
    config, summary, train, valid, test, _, digest = load_and_audit(run_dir, graph_dir)
    check(len(train) > 1000, "F 项应使用真实 MovieLens 训练集，而非微型模拟集")
    check(summary['graph_source'] == 'train_only', "图源并非纯训练集")
    with tempfile.TemporaryDirectory(prefix="day14_validation_") as tmp:
        report = run_day14(run_dir, graph_dir, tmp, embedding_dim=16,
                           n_layers=2, batch_size=2048, epochs=1, topk=10,
                           seed=20261031, device="cpu")
        check(report['train_pairs_sha256'] == digest, "图源校验摘要不一致")
        check(report.get('test_result') is None, "Day 14 禁止使用测试集选模型")
        check(Path(report['checkpoint']).is_file(), "没有保存最佳验证模型")
        for name in ('experiment_config.json', 'epoch_metrics.csv', 'run_summary.json'):
            check((Path(tmp) / name).is_file(), f"缺少 {name}")
        valid_result = report['best_valid_result']
        for key in ('recall@10', 'hit@10', 'ndcg@10'):
            check(key in valid_result and math.isfinite(float(valid_result[key])), f"缺少 {key} 验证指标")


def main():
    parser = argparse.ArgumentParser(description="Day 14｜LightGCN 训练验收")
    parser.add_argument('--run-dir', type=Path, help='Day 7 冻结数据划分目录')
    parser.add_argument('--graph-dir', type=Path, help='Day 11 归一化图目录')
    args = parser.parse_args()
    tests = [
        ('A. 训练正交互与合法负采样', 15, test_A_sampling),
        ('B. BPR 与初始 Embedding 正则化', 20, test_B_loss),
        ('C. 小图训练、参数更新和有限梯度', 20, test_C_training),
        ('D. Full Ranking 和训练已见物品过滤', 15, test_D_full_ranking),
        ('E. 孤立节点与保存重载', 10, test_E_boundaries),
    ]
    points = 0
    for title, score, fn in tests:
        try:
            fn()
            points += score
            print(f'[PASS] {title} (+{score})', flush=True)
        except Exception as e:
            print(f'[FAIL] {title}: {e}', flush=True)
            traceback.print_exc(limit=3)
    if args.run_dir and args.graph_dir:
        try:
            test_F_movie_lens(args.run_dir, args.graph_dir)
            points += 20
            print('[PASS] F. 冻结 MovieLens 训练图集成 (+20)', flush=True)
        except Exception as e:
            print(f'[FAIL] F. 冻结 MovieLens 训练图集成: {e}', flush=True)
            traceback.print_exc(limit=4)
    elif args.run_dir or args.graph_dir:
        print('[FAIL] F 项必须同时提供 --run-dir 和 --graph-dir')
    else:
        print('[SKIP] F. 完整数据集集成 (+20)：需要 --run-dir 和 --graph-dir')
    print(f'Day 14 得分：{points}/100', flush=True)
    if points < (100 if args.run_dir and args.graph_dir else 80):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
