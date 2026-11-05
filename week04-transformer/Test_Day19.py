"""Day 19 自动验收：A–E 代码 90 分，F 理论文档 10 分。

依赖：已经完成的 Day 17/18 attention 模块（同一目录）。
测试只使用 PyTorch 独立计算参考值；不得把官方 nn.TransformerEncoderLayer 当作学生实现。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import traceback
import torch
import torch.nn.functional as F
from torch import nn
from transformer_layers import (
    PostLayerNormResidual,
    PositionwiseFeedForward,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TokenOutputProjection,
)

ROOT = Path(__file__).resolve().parent


def assert_close(a: torch.Tensor, b: torch.Tensor, atol=1e-5, rtol=1e-5):
    assert tuple(a.shape) == tuple(b.shape), f"形状不同：{tuple(a.shape)} != {tuple(b.shape)}"
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def check_finite_grads(mod: nn.Module, *, require_all: bool = True):
    grads = [(n, p.grad) for n, p in mod.named_parameters() if p.requires_grad]
    assert grads, "没有找到可训练参数"
    assert all(g is not None for n, g in grads), f"部分参数没有梯度：{[(n, g is None) for n, g in grads]}"
    assert all(torch.isfinite(g).all() for n, g in grads), "梯度包含 NaN 或 Inf"
    if require_all:
        assert all(g.abs().sum().item() > 0 for n, g in grads), "有参数梯度全零"


def test_a1():
    # 不使用 LayerNorm 对照包装，直接计算均值、方差、仿射变换。
    torch.manual_seed(19)
    m = PostLayerNormResidual(4, dropout_p=0.0).eval()
    x = torch.tensor([[[1., 2., 3., 4.], [1., 1., 3., 7.]]])
    s = torch.tensor([[[1., 0., -1., 0.], [2., 0., 0., -2.]]])
    with torch.no_grad():
        m.norm.weight.copy_(torch.tensor([1.0, 1.5, 0.7, 2.0]))
        m.norm.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.]))
    z = x + s
    expected = (z - z.mean(-1, keepdim=True)) / torch.sqrt(z.var(-1, unbiased=False, keepdim=True) + 1e-5)
    expected = expected * m.norm.weight + m.norm.bias
    assert_close(m(x, s), expected)
    assert len(list(m.norm.parameters())) == 2, "LayerNorm 应有可训练 weight / bias"
    assert not torch.allclose(m(x, s), x + s), "不能省略 LayerNorm"


def test_a2():
    m = PostLayerNormResidual(4, dropout_p=0.0).eval()
    x = torch.tensor([[[1., 2., 4., 8.]]])
    # 对比真正的 Pre-LN F(LN(x)) + x，使用 F(v)=2v；数值必须不同。
    got = m(x, 2*x)
    pre = x + 2*m.norm(x)
    assert not torch.allclose(got, pre, atol=1e-5, rtol=1e-5), "你可能把 Post-LN 误写成了 Pre-LN"
    for bad in [0, -4]:
        try: PostLayerNormResidual(bad)
        except (ValueError, TypeError): pass
        else: raise AssertionError("非法 d_model 需要报错")
    try: m(x, x[:, :, :2])
    except (ValueError, RuntimeError, AssertionError): pass
    else: raise AssertionError("残差相加的两个输入 shape 必须相同")


def test_b1():
    torch.manual_seed(20)
    ffn = PositionwiseFeedForward(8, 16).eval()
    x = torch.randn(2, 4, 8)
    y = ffn(x)
    expected = ffn.linear2(F.relu(ffn.linear1(x)))
    assert_close(y, expected)
    assert y.shape == x.shape
    assert sum(p.numel() for p in ffn.parameters()) == (8*16+16)+(16*8+8), "FFN 参数量不正确"


def test_b2():
    torch.manual_seed(21)
    ffn = PositionwiseFeedForward(8, 16).eval()
    x = torch.randn(2, 4, 8)
    changed = x.clone(); changed[:, 2, :] += 100.0
    y, y_changed = ffn(x), ffn(changed)
    assert_close(y[:, [0,1,3]], y_changed[:, [0,1,3]])
    assert not torch.allclose(y[:, 2], y_changed[:, 2]), "FFN 应对改变的位置产生响应"
    inp = x.clone().requires_grad_(True)
    ffn(inp).square().sum().backward()
    check_finite_grads(ffn)
    assert inp.grad is not None and torch.isfinite(inp.grad).all()


def test_c1():
    torch.manual_seed(22)
    model = TransformerEncoderLayer(8, 2, 16, dropout_p=0.0).eval()
    x = torch.randn(2, 5, 8)
    mask = torch.ones(2, 1, 5, dtype=torch.bool)
    mask[:, :, -1] = False
    y = model(x, src_keep_mask=mask)
    # 直接按论文次序手动计算，检验每一个 Add & Norm 位置。
    first = model.addnorm1(x, model.self_attn(x, keep_mask=mask))
    expected = model.addnorm2(first, model.ffn(first))
    assert_close(y, expected)
    changed = x.clone(); changed[:, -1, :] += 75
    assert_close(y[:, :-1], model(changed, src_keep_mask=mask)[:, :-1])
    assert y.shape == x.shape


def test_c2():
    torch.manual_seed(23)
    model = TransformerEncoderLayer(8, 2, 16, dropout_p=0.0).train()
    x = torch.randn(2, 3, 8, requires_grad=True)
    # 带有非均匀上游梯度，避免 LN 纯常量和的退化情况。
    (model(x) * torch.randn(2, 3, 8)).sum().backward()
    check_finite_grads(model, require_all=False)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert sum(1 for n, _ in model.named_modules() if isinstance(_, PostLayerNormResidual)) == 2


def test_d1():
    torch.manual_seed(24)
    model = TransformerDecoderLayer(8, 2, 16, dropout_p=0.0).eval()
    x = torch.randn(2, 4, 8)
    memory = torch.randn(2, 6, 8)
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    mem_mask = torch.ones(2, 1, 6, dtype=torch.bool); mem_mask[:, :, -1] = False
    y = model(x, memory, causal, mem_mask)
    a = model.addnorm1(x, model.self_attn(x, keep_mask=causal))
    b = model.addnorm2(a, model.cross_attn(a, memory, memory, keep_mask=mem_mask))
    expected = model.addnorm3(b, model.ffn(b))
    assert y.shape == (2, 4, 8)
    assert_close(y, expected)
    changed = x.clone(); changed[:, 3] += 40.0
    assert_close(y[:, :3], model(changed, memory, causal, mem_mask)[:, :3])
    memory_changed = memory.clone(); memory_changed[:, -1] += 100
    assert_close(y, model(x, memory_changed, causal, mem_mask))
    assert sum(1 for n, _ in model.named_modules() if isinstance(_, PostLayerNormResidual)) == 3


def test_d2():
    torch.manual_seed(25)
    model = TransformerDecoderLayer(8, 2, 16, dropout_p=0.0).train()
    x = torch.randn(2, 4, 8, requires_grad=True)
    mem = torch.randn(2, 5, 8, requires_grad=True)
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    out = model(x, mem, causal)
    # Cross Attention 必须实际依赖 Encoder 的 memory。
    mem_changed = mem.detach().clone(); mem_changed[:, 0] += 10.0
    assert not torch.allclose(out.detach(), model(x.detach(), mem_changed, causal)), "Decoder 没有利用 memory"
    (out * torch.randn_like(out)).sum().backward()
    check_finite_grads(model, require_all=False)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert mem.grad is not None and torch.isfinite(mem.grad).all()
    assert mem.grad.abs().sum() > 0, "Encoder memory 没有收到 Cross-Attention 梯度"


def test_e1():
    torch.manual_seed(26)
    proj = TokenOutputProjection(8, 11)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    logits = proj(hidden)
    assert logits.shape == (2, 4, 11)
    assert_close(logits, proj.proj(hidden))
    assert sum(p.numel() for p in proj.parameters()) == 8*11+11
    probs = proj.probabilities(hidden)
    assert_close(probs, torch.softmax(logits, dim=-1))
    assert_close(probs.sum(-1), torch.ones(2, 4))
    assert not torch.allclose(logits, probs), "forward 应返回 logits，不应提前 Softmax"
    targets = torch.tensor([[1,2,3,0],[4,5,6,0]])
    loss = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1), ignore_index=0)
    assert torch.isfinite(loss)
    loss.backward()
    check_finite_grads(proj)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_e2():
    torch.manual_seed(27)
    model = TransformerDecoderLayer(8, 2, 16, dropout_p=0.0).eval()
    out = TokenOutputProjection(8, 13).eval()
    decoder = torch.randn(2, 4, 8)
    memory = torch.randn(2, 6, 8)
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    hidden = model(decoder, memory, causal)
    logits = out(hidden)
    assert logits.shape == (2, 4, 13)
    changed = decoder.clone(); changed[:, 3] += 100
    assert_close(logits[:, :3], out(model(changed, memory, causal))[:, :3])
    try: TokenOutputProjection(8, 0)
    except (ValueError, TypeError): pass
    else: raise AssertionError("非法 vocab_size 应报错")


def test_f_docs():
    paths = [ROOT / "Day19_notes.md", ROOT / "day19_structure.md"]
    for p in paths:
        assert p.exists(), f"缺失阅读作业 {p.name}"
        text = p.read_text(encoding="utf-8")
        assert "【待填写】" not in text, f"请完成 {p.name} 中的全部待填写内容"
        assert len(text.strip()) > 500, f"{p.name} 需要完整的解释和结构图"
    structure = paths[1].read_text(encoding="utf-8")
    assert "Cross-Attention" in structure and "Add & Norm" in structure, "结构图应包含 Cross-Attention 和 Add & Norm"


GROUPS = [
    ("A", 15, [test_a1, test_a2]),
    ("B", 20, [test_b1, test_b2]),
    ("C", 20, [test_c1, test_c2]),
    ("D", 20, [test_d1, test_d2]),
    ("E", 15, [test_e1, test_e2]),
    ("F", 10, [test_f_docs]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-docs", action="store_true", help="暂时跳过 F 组理论文件")
    parser.add_argument("--verbose", action="store_true", help="打印详细 Traceback")
    args = parser.parse_args()
    total = 0
    for name, score, tests in GROUPS:
        if name == "F" and args.skip_docs:
            print("[SKIP] F. 理论笔记与结构图（10 分）")
            continue
        try:
            for test in tests:
                test()
            total += score
            print(f"[PASS] {name}. +{score} 分")
        except Exception as exc:
            print(f"[FAIL] {name}. +0/{score} 分：{type(exc).__name__}: {exc}")
            if args.verbose:
                traceback.print_exc()
    print(f"\nDay 19 总分：{total}/100" + ("（理论文件未计分）" if args.skip_docs else ""))
    if total != (90 if args.skip_docs else 100):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
