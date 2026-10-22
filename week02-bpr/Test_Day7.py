"""第 7 天修订版自动测试：LearnBPR SGD 与自助采样，对照 Adam。
在包含四个 .py 文件的目录运行：
  python Test_Day7.py
  python Test_Day7.py --ml100k data/ml-100k/u.data
A-E 使用合成数据，占 80 分；F 使用官方 MovieLens-100K，占 20 分。不含答案，笔记和理论要求需手工验收。"""
from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from bpr_model import BPRMF, bpr_loss
from negative_sampling import build_positive_sets, BootstrapTripletSampler
from train import load_ml100k_ratings, split_ratings, synthetic_pairs, train_one_epoch, run_experiment


def ensure(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def near(got, wanted, message, atol=1e-6):
    ensure(torch.allclose(torch.as_tensor(got), torch.as_tensor(wanted), atol=atol, rtol=1e-6),
           message + f': got {got}; expected {wanted}')


def test_a_model_loss():
    torch.manual_seed(8)
    model = BPRMF(3, 4, 2)
    ensure(isinstance(model.user_embedding, torch.nn.Embedding), 'user_embedding 应为 nn.Embedding')
    ensure(isinstance(model.item_embedding, torch.nn.Embedding), 'item_embedding 应为 nn.Embedding')
    ensure(model.user_embedding.weight.shape == (3, 2), '用户 Embedding 维度错误')
    ensure(model.item_embedding.weight.shape == (4, 2), '物品 Embedding 维度错误')
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1., 2.], [3., -1.], [2., 2.]]))
        model.item_embedding.weight.copy_(torch.tensor([[1., 0.], [0., 2.], [2., -1.], [-1., 1.]]))
    u = torch.tensor([0, 1, 1, 2]); i = torch.tensor([0, 1, 2, 3]); j = torch.tensor([2, 3, 0, 0])
    pos, neg = model(u, i, j)
    ensure(pos.shape == (4,) and neg.shape == (4,), 'forward 应输出两个 [B]')
    near(pos, torch.tensor([1., -2., 7., 0.]), '正物品分数错误')
    near(neg, torch.tensor([0., -4., 3., 2.]), '负物品分数错误')
    ensure(model.score(torch.tensor([0]), torch.tensor([1])).shape == (1,), 'Batch=1 应为 [1]')
    near(bpr_loss(torch.zeros(1), torch.zeros(1)), torch.tensor(math.log(2.)), 'Δ=0 时 Loss=ln2')
    huge = bpr_loss(torch.tensor([-1000., 1000.]), torch.tensor([1000., -1000.]))
    ensure(torch.isfinite(huge), 'Loss 应具有数值稳定性')
    pa = torch.tensor([0.5], requires_grad=True); na = torch.tensor([-0.5], requires_grad=True)
    bpr_loss(pa, na).backward()
    ensure(pa.grad.item() < 0 and na.grad.item() > 0, '正负分数的梯度方向错误')


def test_b_bootstrap_uniform():
    train_pairs = [(0, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]
    ps = build_positive_sets(train_pairs, n_users=4)
    ensure(ps == {0: {0}, 1: {1, 2}, 2: {0, 1, 2, 3}, 3: set()}, '训练正样本集合错误')
    sampler = BootstrapTripletSampler(ps, n_items=4)
    ensure(sampler.num_legal_triplets == 7, '三元组个数应为 3+4=7')
    x = sampler.sample(7000, random.Random(20261020))
    y = sampler.sample(7000, random.Random(20261020))
    ensure(x == y, 'Bootstrap 固定随机种子须复现同样顺序')
    ensure(len(x) == 7000 and len(set(x)) < len(x), '必须有放回抽样，允许重复三元组')
    ensure(all(u in (0, 1) and i in ps[u] and j not in ps[u] and 0 <= j < 4 for u, i, j in x),
           '采样包含非法三元组（包括取到训练正样本）')
    ratio_u1 = sum(u == 1 for u, i, j in x) / len(x)
    ensure(abs(ratio_u1 - 4 / 7) < 0.035,
           f'必须均匀采合法三元组：用户 1 理论频率 4/7，得到 {ratio_u1:.3f}；每正例采一次不满足这一条件')
    ensure(sampler.sample(0, random.Random(1)) == [], '抽样 0 次应返回 []')
    for bad_n in (-1,):
        try:
            sampler.sample(bad_n, random.Random(1))
        except ValueError:
            pass
        else:
            raise AssertionError('负抽样次数应抛 ValueError')
    empty = BootstrapTripletSampler({0: set(), 1: {0, 1, 2}}, 3)
    ensure(empty.num_legal_triplets == 0, '无合法三元组时个数应为 0')
    try:
        empty.sample(1, random.Random(2))
    except ValueError:
        pass
    else:
        raise AssertionError('无合法三元组不能死循环，应直接 ValueError')


def test_c_sgd_step_exact():
    """单个三元组，严格对照普通 SGD 的一次梯度更新，阻止暗中使用 Adam。"""
    torch.manual_seed(3)
    m = BPRMF(1, 2, 2)
    with torch.no_grad():
        m.user_embedding.weight.copy_(torch.tensor([[1.0, 1.0]]))
        m.item_embedding.weight.copy_(torch.tensor([[0.3, 0.2], [-0.2, 0.4]]))
    original = {k: v.detach().clone() for k, v in m.state_dict().items()}
    u = torch.tensor([0]); i = torch.tensor([0]); j = torch.tensor([1])
    pos, neg = m(u, i, j)
    ref_loss = F.softplus(neg - pos).mean()
    grads = torch.autograd.grad(ref_loss, tuple(m.parameters()))
    expected = {k: original[k] - 0.1 * grad for k, grad in zip(original, grads)}
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    ps = build_positive_sets([(0, 0)], n_users=1)
    sampler = BootstrapTripletSampler(ps, n_items=2)
    stats = train_one_epoch(m, sampler, opt, batch_size=1, steps_per_epoch=1,
                            rng=random.Random(0), l2_weight=0.0)
    ensure(stats['num_triplets'] == 1, '应仅训练一个三元组')
    for name, tensor in m.state_dict().items():
        near(tensor, expected[name], f'SGD 单步更新不正确: {name}', atol=1e-6)
    ensure(math.isfinite(float(stats['loss'])), '训练指标应为有限值')


def fixed_rank_metrics(m, pairs, positives, n_items):
    triples = [(u, i, j) for u, i in pairs for j in range(n_items) if j not in positives[u]]
    tt = torch.tensor(triples, dtype=torch.long)
    with torch.no_grad():
        a, b = m(tt[:, 0], tt[:, 1], tt[:, 2])
        return float((a > b).float().mean()), float(bpr_loss(a, b))


def test_d_sgd_convergence():
    torch.manual_seed(20261020)
    train_pairs, n_users, n_items = synthetic_pairs()
    ensure((n_users, n_items, len(train_pairs)) == (6, 15, 18), '合成数据应为 6×15、18 个正交互')
    positives = build_positive_sets(train_pairs, n_users)
    sampler = BootstrapTripletSampler(positives, n_items)
    m = BPRMF(n_users, n_items, 12)
    optimizer = torch.optim.SGD(m.parameters(), lr=0.20)
    before_acc, before_loss = fixed_rank_metrics(m, train_pairs, positives, n_items)
    for epoch in range(50):
        s = train_one_epoch(m, sampler, optimizer, batch_size=8, steps_per_epoch=216,
                            rng=random.Random(20261020 + epoch), l2_weight=1e-5)
        for key in ('loss', 'bpr_loss', 'pairwise_acc', 'num_triplets', 'num_unique_triplets', 'sample_sha256'):
            ensure(key in s, f'缺少统计项 {key}')
        ensure(s['num_triplets'] == 216, '每个 Epoch 的抽样步数不正确')
        ensure(1 <= s['num_unique_triplets'] < 216, '有放回训练应包含重复抽到的三元组')
        ensure(0 <= s['pairwise_acc'] <= 1, '成对正确率应位于 [0,1]')
        ensure(math.isfinite(float(s['loss'])) and math.isfinite(float(s['bpr_loss'])), 'Loss 必须有限')
    after_acc, after_loss = fixed_rank_metrics(m, train_pairs, positives, n_items)
    ensure(after_acc >= 0.90, f'SGD 合成数据固定三元组正确率需 >= 90%，当前 {after_acc:.3f}')
    ensure(after_loss < 0.5 * before_loss,
           f'SGD 合成数据固定三元组 Loss 下降不足：{before_loss:.3f} -> {after_loss:.3f}')
    print(f'    SGD 合成数据：固定三元组 accuracy {before_acc:.3f} -> {after_acc:.3f}；'
          f'BPR Loss {before_loss:.3f} -> {after_loss:.3f}')


def assert_run_artifacts(out, expected_epochs):
    for name in ('config.json', 'train_pairs.tsv', 'valid_pairs.tsv', 'test_pairs.tsv',
                 'history.jsonl', 'checkpoint.pt'):
        ensure((out / name).is_file(), f'缺少实验文件 {name}')
    conf = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    hist = [json.loads(x) for x in (out / 'history.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    ck = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=True)
    ensure(len(hist) == expected_epochs, 'history.jsonl 每个 Epoch 应有一行')
    ensure('state_dict' in ck and 'optimizer_state_dict' in ck and 'config' in ck,
           'Checkpoint 需要模型、优化器和实验配置')
    ensure(conf['sampler'] == 'bootstrap_uniform_with_replacement', '采样协议必须明确为均匀有放回')
    ensure('initial_weight_sha256' in conf and len(conf['initial_weight_sha256']) == 64,
           '缺少初始参数摘要 SHA256')
    ensure(all(len(h['sample_sha256']) == 64 for h in hist), '每轮需要三元组抽样 SHA256')
    return conf, hist, ck


def test_e_adam_control():
    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td) / 'sgd'
        adir = Path(td) / 'adam'
        args = dict(dataset='synthetic', data_path=None, epochs=3, seed=20261020,
                    embedding_dim=8, batch_size=8, steps_per_epoch=72, l2_weight=0.0)
        run_experiment(output_dir=sdir, optimizer_name='sgd', lr=0.2, **args)
        run_experiment(output_dir=adir, optimizer_name='adam', lr=0.02, **args)
        sc, sh, sk = assert_run_artifacts(sdir, 3)
        ac, ah, ak = assert_run_artifacts(adir, 3)
        ensure(sc['optimizer'] == 'sgd' and ac['optimizer'] == 'adam', '配置中应正确记录优化器')
        ensure(sc['initial_weight_sha256'] == ac['initial_weight_sha256'], '对照必须使用相同初始权重')
        ensure([h['sample_sha256'] for h in sh] == [h['sample_sha256'] for h in ah],
               '对照必须使用完全相同的随机三元组序列')
        for name in ('train_pairs.tsv', 'valid_pairs.tsv', 'test_pairs.tsv'):
            ensure((sdir / name).read_bytes() == (adir / name).read_bytes(), '对照划分不一致')
        ensure(sk['optimizer_state_dict']['state'] == {}, '无动量 SGD 的状态表应为空；疑似没有实际使用 SGD')
        ensure(bool(ak['optimizer_state_dict']['state']), 'Adam 必须维护状态，不能仅在配置里声明 Adam')
        ensure(any(not torch.allclose(sk['state_dict'][k], ak['state_dict'][k])
                   for k in sk['state_dict']), '两个优化器结束后权重不应完全一致')
        ensure(all(math.isfinite(float(h['loss'])) for h in sh + ah), '两组训练 Loss 应有限')


def test_f_ml100k(path):
    ratings = load_ml100k_ratings(path)
    ensure(len(ratings) == 100000, f'MovieLens-100K u.data 应有 100000 行，实际 {len(ratings)}')
    ensure(min(u for u, _, _, _ in ratings) == 0 and max(u for u, _, _, _ in ratings) == 942,
           'MovieLens 用户 ID 应为零基 0...942')
    ensure(min(i for _, i, _, _ in ratings) == 0 and max(i for _, i, _, _ in ratings) == 1681,
           'MovieLens 物品 ID 应为零基 0...1681')
    ensure(all(1 <= r <= 5 for _, _, r, _ in ratings), '评分范围应是 1...5')
    tr, va, te = split_ratings(ratings, .1, .1, 20261020)
    tr2, va2, te2 = split_ratings(ratings, .1, .1, 20261020)
    ensure(tr == tr2 and va == va2 and te == te2, '数据划分必须可复现')
    ensure(tr and va and te, '训练/验证/测试均不可为空')
    ensure(not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te)),
           '三份交互不允许交叉')
    ensure(len(set(tr) | set(va) | set(te)) == 100000, '三份交互必须完整覆盖去重交互')
    ensure(len({u for u, _ in tr}) == 943, '训练集每位 MovieLens 用户至少一条正交互')
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        run_experiment('ml100k', path, out, epochs=1, seed=20261020, embedding_dim=8,
                       batch_size=512, lr=0.05, optimizer_name='sgd')
        config, history, ck = assert_run_artifacts(out, 1)
        ensure(config['optimizer'] == 'sgd', 'MovieLens 主实验必须是 SGD')
        ensure(config['n_users'] == 943 and config['n_items'] == 1682, '模型实体数量错误')
        ensure((config['n_train'], config['n_valid'], config['n_test']) ==
               (len(tr), len(va), len(te)), '保存的三份交互数错误')
        for name, expected in (('train_pairs.tsv', tr), ('valid_pairs.tsv', va), ('test_pairs.tsv', te)):
            observed = {tuple(map(int, line.split('\t'))) for line in
                        (out / name).read_text(encoding='utf-8').splitlines() if line.strip()}
            ensure(observed == set(expected), f'{name} 内容与实际划分不符')
        ensure(history[0]['num_triplets'] == len(tr), '默认一轮抽样次数应等于训练正交互数')
        ensure(math.isfinite(history[0]['loss']), 'MovieLens-100K 训练 Loss 非有限数')
        print(f'    MovieLens: train={len(tr)}, valid={len(va)}, test={len(te)}, epoch=1')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ml100k', default=None, help='官方 MovieLens-100K 解压得到的 u.data')
    p.add_argument('--verbose', action='store_true', help='输出测试异常堆栈')
    args = p.parse_args()
    tests = [
        ('A. MF 分数 / BPR Loss / 梯度', 15, test_a_model_loss),
        ('B. 三元组均匀 Bootstrap 与边界', 20, test_b_bootstrap_uniform),
        ('C. SGD 单步参数更新（排除 Adam 偷换）', 20, test_c_sgd_step_exact),
        ('D. SGD + Bootstrap 合成数据收敛', 15, test_d_sgd_convergence),
        ('E. Adam 对照：相同初始化 / 相同采样', 10, test_e_adam_control),
    ]
    score = passed = 0
    for label, points, fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f'[FAIL] {label}: {type(exc).__name__}: {exc}')
            if args.verbose:
                traceback.print_exc()
        else:
            score += points
            passed += 1
            print(f'[PASS] {label} (+{points})')
    if args.ml100k:
        try:
            test_f_ml100k(Path(args.ml100k))
        except Exception as exc:
            print(f'[FAIL] F. MovieLens-100K SGD 一轮训练: {type(exc).__name__}: {exc}')
            if args.verbose:
                traceback.print_exc()
        else:
            score += 20
            passed += 1
            print('[PASS] F. MovieLens-100K SGD 一轮训练 (+20)')
    else:
        print('[SKIP] F. MovieLens-100K SGD 一轮训练 (+20)，未提供 --ml100k')
    print(f'\n新版 Day 7 自动验收：{passed}/6 项通过，得分 {score}/100')
    if score < 100:
        print('说明：没有真实 MovieLens 文件时，最多 80 分；理论笔记仍需独立验收。')
    raise SystemExit(0 if score == 100 else 1)


if __name__ == '__main__':
    main()
