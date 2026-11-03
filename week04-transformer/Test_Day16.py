"""Day 16 自动验收（100 分）。

A：Token Embedding（15）
B：正弦位置编码（25）
C：输入融合与 Padding（20）
D：Decoder Shifted-Right（20）
E：梯度与边界（10）
F：论文笔记与架构图（10）

使用方法：python Test_Day16.py
           python Test_Day16.py --project-dir . --skip-docs

只包含测试与检查标准，不提供完整参考实现。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import traceback
import torch

from embeddings_position import (
    TokenEmbedding, SinusoidalPositionalEncoding, TransformerInputEmbedding,
    make_valid_mask, shift_right,
)


def close(actual, expected, atol=1e-6):
    assert actual.shape == expected.shape, f"形状不符: {actual.shape} != {expected.shape}"
    assert torch.allclose(actual, expected, atol=atol, rtol=1e-6), (
        f"数值不符: max_err={(actual-expected).abs().max().item()}"
    )


def reference_pos(position: int, dim: int):
    # 这是一个具体位置的独立标量计算，用于核对数学公式，而非完整实现。
    assert dim % 2 == 0
    result = []
    for j in range(dim):
        angle = position / (10000.0 ** ((2 * (j // 2)) / dim))
        result.append(math.sin(angle) if j % 2 == 0 else math.cos(angle))
    return torch.tensor(result, dtype=torch.float32)


def test_a1():
    torch.manual_seed(12)
    m = TokenEmbedding(13, 8, padding_idx=0)
    assert isinstance(m.embedding, torch.nn.Embedding)
    assert m.embedding.weight.shape == (13, 8)
    ids = torch.tensor([[1, 2, 3], [3, 0, 4]], dtype=torch.long)
    out = m(ids)
    assert out.shape == (2, 3, 8)
    close(out, m.embedding(ids) * math.sqrt(8))


def test_a2():
    m = TokenEmbedding(9, 6, padding_idx=0)
    ids = torch.tensor([[1, 2, 0]], dtype=torch.long)
    assert m.embedding.padding_idx == 0
    close(m.embedding.weight[0].detach(), torch.zeros(6))
    loss = m(ids).square().sum()
    loss.backward()
    assert m.embedding.weight.grad is not None
    close(m.embedding.weight.grad[0], torch.zeros(6))
    assert m.embedding.weight.grad[1].norm().item() > 0


def test_b1():
    m = SinusoidalPositionalEncoding(8, max_len=32)
    assert hasattr(m, "pe")
    assert m.pe.shape == (32, 8)
    assert "pe" in dict(m.named_buffers()), "位置编码必须注册为 Buffer"
    assert "pe" not in dict(m.named_parameters()), "位置编码不能成为独立训练参数"
    for pos in (0, 1, 4, 7):
        close(m.pe[pos], reference_pos(pos, 8))
    close(m.pe[0, ::2], torch.zeros(4))
    close(m.pe[0, 1::2], torch.ones(4))


def test_b2():
    m = SinusoidalPositionalEncoding(8, max_len=32)
    x = torch.zeros(3, 4, 8)
    out = m(x, start_pos=2)
    assert out.shape == x.shape
    for i in range(4):
        close(out[0, i], reference_pos(2 + i, 8))
    close(out[0], out[2])
    assert not torch.allclose(out[:, 0], out[:, 1])


def test_b3():
    m = SinusoidalPositionalEncoding(8, max_len=10)
    x = torch.ones(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    out = m(x, valid_mask=mask)
    close(out[0, 2], torch.zeros(8))
    close(out[1, 1], torch.zeros(8))
    close(out[1, 0], x[1, 0] + reference_pos(0, 8))


def test_c1():
    ids = torch.tensor([[1, 2, 0, 0], [3, 2, 4, 0]])
    mask = make_valid_mask(ids, padding_idx=0)
    assert mask.dtype == torch.bool and mask.shape == ids.shape
    close(mask.to(torch.int32), (ids != 0).to(torch.int32))
    m = TransformerInputEmbedding(15, 8, max_len=20)
    assert hasattr(m, "token_embedding") and hasattr(m, "position_encoding")
    out = m(ids)
    assert out.shape == (2, 4, 8)
    close(out[0, 2:], torch.zeros(2, 8))
    close(out[1, 3], torch.zeros(8))


def test_c2():
    torch.manual_seed(5)
    m = TransformerInputEmbedding(15, 8, max_len=20)
    ids = torch.tensor([[4, 5, 6], [4, 6, 0]])
    out = m(ids)
    tok = m.token_embedding(ids)
    ref = m.position_encoding(tok, valid_mask=(ids != 0))
    close(out, ref)
    # 同一 Token 在不同位置：查表部分相同，位置编码不同
    repeated = torch.tensor([[3, 3, 3]])
    repeated_out = m(repeated)
    assert not torch.allclose(repeated_out[0, 0], repeated_out[0, 1])


def test_d1():
    tgt = torch.tensor([[5, 7, 9, 2, 0], [8, 6, 2, 0, 0]], dtype=torch.long)
    out = shift_right(tgt, bos_id=1, padding_idx=0)
    expected = torch.tensor([[1, 5, 7, 9, 2], [1, 8, 6, 2, 0]], dtype=torch.long)
    assert torch.equal(out, expected), f"Shifted-Right 错误: {out}"
    assert out.shape == tgt.shape and out.dtype == torch.long


def test_d2():
    for seq in (torch.tensor([[4]]), torch.tensor([[4, 5, 2]]), torch.tensor([[0, 0, 0]])):
        out = shift_right(seq, bos_id=1, padding_idx=0)
        assert out[0, 0].item() == 1
        assert torch.equal(out[:, 1:], seq[:, :-1])
        # 任意 t 位置不能将其 target[t] 直接拷贝为同位置输入
        if seq.shape[1] > 1:
            assert out.shape == seq.shape


def test_e1():
    m = TransformerInputEmbedding(16, 8, max_len=12)
    x = torch.tensor([[2, 4, 6, 0]], dtype=torch.long)
    y = m(x)
    (y.square().sum()).backward()
    grad = m.token_embedding.embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad[2].abs().sum().item() > 0
    close(grad[0], torch.zeros(8))
    assert len([p for p in m.parameters() if p.requires_grad]) == 1, "PE 不应训练"


def must_raise(fn, why):
    try:
        fn()
    except (ValueError, TypeError, IndexError, AssertionError):
        return
    raise AssertionError(f"应当对非法输入报错：{why}")


def test_e2():
    m = SinusoidalPositionalEncoding(8, max_len=5)
    must_raise(lambda: m(torch.zeros(2, 6, 8)), "序列长度超出 max_len")
    must_raise(lambda: m(torch.zeros(2, 3, 8), start_pos=4), "位置偏移越界")
    must_raise(lambda: SinusoidalPositionalEncoding(5), "本练习要求偶数 d_model")
    m2 = TokenEmbedding(10, 8)
    must_raise(lambda: m2(torch.tensor([[1.0, 2.0]])), "ID 必须为整数")
    must_raise(lambda: shift_right(torch.tensor([], dtype=torch.long).view(1, 0), bos_id=1), "空目标序列")


def test_f1(project_dir):
    file = project_dir / "transformer_architecture.md"
    assert file.is_file(), "缺少 transformer_architecture.md"
    content = file.read_text(encoding="utf-8")
    assert "【待填写】" not in content, "请先完成架构图中的待填写部分"
    assert len(content) >= 450, "架构图及说明不足 450 个字符"
    assert "Encoder" in content and "Decoder" in content
    assert "Shifted" in content or "右移" in content


def test_f2(project_dir):
    file = project_dir / "Day16_notes.md"
    assert file.is_file(), "缺少 Day16_notes.md"
    content = file.read_text(encoding="utf-8")
    assert "【待填写】" not in content, "请先完成学习笔记中的待填写部分"
    assert len(content) >= 550, "笔记内容不足 550 个字符"
    assert "Q" in content and "K" in content and "V" in content
    assert "SASRec" in content


def main():
    parser = argparse.ArgumentParser(description="Day 16 自动评分（100 分）")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--skip-docs", action="store_true", help="跳过 F 文档项，分数上限 90")
    args = parser.parse_args()
    cases = [
        ("A1 查表、维度与缩放", 10, test_a1),
        ("A2 Padding 行梯度", 5, test_a2),
        ("B1 PE 公式与 Buffer", 10, test_b1),
        ("B2 位置偏移与 Batch", 10, test_b2),
        ("B3 PE Padding 处理", 5, test_b3),
        ("C1 Mask 与融合输出", 10, test_c1),
        ("C2 相同 Token 的位置区别", 10, test_c2),
        ("D1 Shifted-Right 标准样例", 10, test_d1),
        ("D2 Shifted-Right 边界情况", 10, test_d2),
        ("E1 梯度与参数审计", 5, test_e1),
        ("E2 非法输入和边界", 5, test_e2),
    ]
    if not args.skip_docs:
        cases.extend([
            ("F1 Encoder/Decoder 架构图", 5, lambda: test_f1(args.project_dir)),
            ("F2 学习笔记", 5, lambda: test_f2(args.project_dir)),
        ])
    earned = 0
    for label, pts, func in cases:
        try:
            func()
            earned += pts
            print(f"[PASS] {label} (+{pts})")
        except Exception as e:
            print(f"[FAIL] {label} (+0/{pts})：{type(e).__name__}: {e}")
    print(f"\nDay 16 总分：{earned}/100")
    if args.skip_docs:
        print("本次跳过 F 文档考核；最高得分为 90/100。")
    if earned < (90 if args.skip_docs else 100):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
