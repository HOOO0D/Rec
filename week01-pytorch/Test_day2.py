"""Day2 验收器：无需修改断言。python Test_day2.py [A|B|C|D|E]
未完成 TODO 时失败是预期行为；全部通过才完成代码验收。勿使用 python -O。
"""
import sys
import torch
from torch import nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import embedding as student

ROWS = [(0, 1, 1.), (1, 2, 2.), (2, 0, 3.), (3, 3, 4.),
        (0, 4, 5.), (1, 0, 6.), (2, 2, 7.)]


def check_A():
    table = student.make_embedding(5, 3)
    assert isinstance(table, nn.Embedding)
    assert table.weight.shape == (5, 3) and table.weight.requires_grad
    assert any(p is table.weight for p in table.parameters())
    with torch.no_grad():
        table.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))
    for ids in (torch.tensor([0, 3, 3]), torch.tensor([4]), torch.tensor([[1, 2], [0, 4]])):
        out = student.lookup(table, ids)
        assert out.shape == (*ids.shape, 3)
        torch.testing.assert_close(out, table.weight[ids])
    print('[PASS] A: 参数表、重复ID、查表形状')


def check_B():
    ds = student.RatingDataset(ROWS)
    assert isinstance(ds, Dataset) and len(ds) == 7
    for index, expected in enumerate(ROWS):
        sample = ds[index]
        assert len(sample) == 3
        for value, dtype, answer in zip(sample, (torch.long, torch.long, torch.float32), expected):
            assert isinstance(value, torch.Tensor) and value.shape == ()
            assert value.dtype == dtype and value.item() == answer
    assert len(student.RatingDataset([])) == 0
    print('[PASS] B: Dataset 三个接口、标量类型、完整数据')


def check_C():
    for shuffle in (False, True):
        loader = student.make_loader(student.RatingDataset(ROWS), 3, shuffle)
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 3 and loader.num_workers == 0 and not loader.drop_last
        seen, sizes = [], []
        for users, items, ratings in loader:
            b = users.shape[0]
            assert users.shape == items.shape == ratings.shape == (b,)
            assert users.dtype == items.dtype == torch.long
            assert ratings.dtype == torch.float32
            sizes.append(b)
            seen.extend(zip(users.tolist(), items.tolist(), ratings.tolist()))
        assert sizes == [3, 3, 1]
        assert Counter(seen) == Counter(ROWS), '不能漏样本或重复采样'
        if not shuffle:
            assert seen == ROWS
    print('[PASS] C: [3,3,1] 尾批、顺序与完整遍历')


def check_D():
    users, items = nn.Embedding(3, 2), nn.Embedding(4, 2)
    with torch.no_grad():
        users.weight.copy_(torch.tensor([[1., 2.], [3., 4.], [-1., 2.]]))
        items.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., -1.]]))
    for u, i, expected in (([0, 1, 2], [2, 3, 1], [3., 2., 2.]), ([1], [0], [3.])):
        scores = student.pair_scores(users, items, torch.tensor(u), torch.tensor(i))
        assert scores.shape == (len(u),) and scores.requires_grad
        torch.testing.assert_close(scores, torch.tensor(expected))
    print('[PASS] D: 成对点积数值、[B] 输出、B=1')


def check_E():
    table = nn.Embedding(5, 2)
    with torch.no_grad():
        table.weight.fill_(1.)
    optimizer = torch.optim.SGD(table.parameters(), lr=0.1)
    ids = torch.tensor([1, 3, 3], dtype=torch.long)
    expected_grad = torch.zeros_like(table.weight)
    expected_grad[1] = 1.
    expected_grad[3] = 2.
    # 连做两次可检查 zero_grad 是否有效；每次须重新前向构建计算图。
    for expected_loss in (6., 5.):
        before = table.weight.detach().clone()
        loss = student.update_once(table, ids, optimizer)
        assert isinstance(loss, float) and abs(loss - expected_loss) < 1e-5
        assert table.weight.grad is not None
        torch.testing.assert_close(table.weight.grad, expected_grad)
        torch.testing.assert_close(table.weight.detach(), before - 0.1 * expected_grad)
        assert torch.equal(table.weight.detach()[[0, 2, 4]], before[[0, 2, 4]])
    print('[PASS] E: 访问行梯度、重复ID累加、更新、未访问行不变、清梯度')


def main():
    if not __debug__:
        raise RuntimeError('不能用 -O 运行：会禁用断言')
    torch.manual_seed(42)
    checks = dict(A=check_A, B=check_B, C=check_C, D=check_D, E=check_E)
    selected = sys.argv[1:]
    if selected and any(key not in checks for key in selected):
        raise SystemExit('用法：python Test_day2.py [A B C D E]')
    failed = []
    for key in selected or checks:
        try:
            checks[key]()
        except Exception as error:
            failed.append(key)
            print(f'[FAIL] {key}: {type(error).__name__}: {error}')
    if failed:
        raise SystemExit(1)
    print('ALL SELECTED TESTS PASSED' if selected else 'ALL DAY2 TESTS PASSED')


if __name__ == '__main__':
    main()
