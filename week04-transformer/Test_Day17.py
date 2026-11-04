"""Day 17 自动验收：A–D 代码 90 分，E 理论笔记 10 分。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import traceback

import torch

from scaled_attention import (
    make_padding_keep_mask,
    make_causal_keep_mask,
    combine_keep_masks,
    scaled_dot_product_attention,
)

ROOT = Path(__file__).resolve().parent


def assert_close(actual, expected, atol=1e-5, rtol=1e-5):
    assert tuple(actual.shape) == tuple(expected.shape), f"形状错误：{tuple(actual.shape)} != {tuple(expected.shape)}"
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_a1():
    # 人工手算：q=[1,0]，k1=[1,0]、k2=[0,1]
    q = torch.tensor([[[1., 0.]]])
    k = torch.tensor([[[1., 0.], [0., 1.]]])
    v = torch.tensor([[[2., 0.], [0., 4.]]])
    out, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    p = math.exp(1/math.sqrt(2)) / (math.exp(1/math.sqrt(2)) + 1)
    expected_weights = torch.tensor([[[p, 1-p]]])
    expected_out = torch.tensor([[[2*p, 4*(1-p)]]])
    assert_close(w, expected_weights)
    assert_close(out, expected_out)
    assert_close(w.sum(-1), torch.ones(1, 1))


def test_a2():
    # Cross-Attention: Query、Key 的序列长度不同，dv 也不同
    torch.manual_seed(17)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 7)
    out, weights = scaled_dot_product_attention(q, k, v, return_weights=True)
    expected_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(4)
    expected_w = torch.softmax(expected_scores, dim=-1)
    assert_close(weights, expected_w)
    assert_close(out, expected_w @ v)
    assert out.shape == (2, 3, 7)


def test_b1():
    tokens = torch.tensor([[5, 7, 0, 0], [4, 9, 2, 0]], dtype=torch.long)
    k = make_padding_keep_mask(tokens)
    q = make_padding_keep_mask(tokens, for_query=True)
    assert k.dtype == torch.bool and k.shape == (2, 1, 4)
    assert q.dtype == torch.bool and q.shape == (2, 4, 1)
    assert_close(k, torch.tensor([[[1,1,0,0]], [[1,1,1,0]]], dtype=torch.bool))
    assert_close(q[..., 0], torch.tensor([[1,1,0,0], [1,1,1,0]], dtype=torch.bool))


def test_b2():
    mask = make_causal_keep_mask(4)
    expected = torch.tensor([
        [1,0,0,0], [1,1,0,0], [1,1,1,0], [1,1,1,1]
    ], dtype=torch.bool)
    assert mask.dtype == torch.bool
    assert_close(mask, expected)
    # Day 20 会用到的非方阵场景；今天提前确认位置含义
    assert_close(make_causal_keep_mask(1, 4, query_start=3), torch.ones(1, 4, dtype=torch.bool))


def test_b3():
    ids = torch.tensor([[5, 7, 0], [8, 9, 10]])
    merged = combine_keep_masks(
        make_padding_keep_mask(ids),
        make_padding_keep_mask(ids, for_query=True),
        make_causal_keep_mask(3),
    )
    expected = torch.tensor([
        [[1,0,0],[1,1,0],[0,0,0]],
        [[1,0,0],[1,1,0],[1,1,1]],
    ], dtype=torch.bool)
    assert merged.dtype == torch.bool and merged.shape == (2,3,3)
    assert_close(merged, expected)
    assert combine_keep_masks(None, None) is None


def test_c1():
    # 修改未来 Token 不能改变前缀输出（Self-Attention Q=K=V）
    x = torch.tensor([[[.2,1.,0.], [0.,1.,.3], [1.,-.7,.2], [.3,.9,-.2]]])
    changed = x.clone()
    changed[:, 3, :] = torch.tensor([100.,-100.,50.])
    causal = make_causal_keep_mask(4)
    y1 = scaled_dot_product_attention(x, x, x, causal)
    y2 = scaled_dot_product_attention(changed, changed, changed, causal)
    assert_close(y1[:, :3], y2[:, :3])


def test_c2():
    # 不加 Causal Mask 时，未来 Token 会影响前缀（判别测试有效性）
    x = torch.tensor([[[1.,0.], [0.,1.], [1.,1.]]])
    x2 = x.clone()
    x2[:,2] = torch.tensor([12.,-7.])
    p = scaled_dot_product_attention(x, x, x)
    q = scaled_dot_product_attention(x2, x2, x2)
    assert torch.max(torch.abs(p[:,:2] - q[:,:2])).item() > 0.05
    causal = make_causal_keep_mask(3)
    p2 = scaled_dot_product_attention(x, x, x, causal)
    q2 = scaled_dot_product_attention(x2, x2, x2, causal)
    assert_close(p2[:,:2], q2[:,:2])


def test_d1():
    # 有效行的被屏蔽权重精确为 0；全屏蔽行全为 0
    torch.manual_seed(123)
    q = torch.randn(1,3,4)
    k = torch.randn(1,3,4)
    v = torch.randn(1,3,2)
    mask = torch.tensor([[[1,0,0],[1,1,0],[0,0,0]]], dtype=torch.bool)
    y, w = scaled_dot_product_attention(q,k,v,mask,return_weights=True)
    assert torch.isfinite(y).all() and torch.isfinite(w).all()
    assert torch.all(w[~mask] == 0)
    assert_close(w.sum(-1), torch.tensor([[1.,1.,0.]]))
    assert_close(y[0,2], torch.zeros(2))
    # 所有 Key 都是 Padding
    all_off = torch.zeros(1,3,3,dtype=torch.bool)
    y2,w2 = scaled_dot_product_attention(q,k,v,all_off,return_weights=True)
    assert_close(y2,torch.zeros_like(y2))
    assert_close(w2,torch.zeros_like(w2))


def test_d2():
    torch.manual_seed(7)
    q = torch.randn(2,3,4,requires_grad=True)
    k = torch.randn(2,4,4,requires_grad=True)
    v = torch.randn(2,4,5,requires_grad=True)
    keep = torch.tensor([[1,1,0,0],[1,1,1,0],[1,1,1,1]], dtype=torch.bool)
    y=scaled_dot_product_attention(q,k,v,keep)
    y.square().sum().backward()
    for name,t in [("Query",q),("Key",k),("Value",v)]:
        assert t.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(t.grad).all(), f"{name} 梯度无穷大或 NaN"
        assert torch.any(t.grad!=0), f"{name} 梯度意外全为零"


def test_d3():
    q=torch.ones(1,2,3)
    k=torch.ones(1,2,3)
    v=torch.ones(1,2,2)
    try:
        combine_keep_masks(torch.ones(2,2,dtype=torch.float))
    except (TypeError,ValueError,AssertionError):
        pass
    else:
        raise AssertionError("不应把浮点 Mask 当作 True=允许的布尔 Mask")
    try:
        scaled_dot_product_attention(q,k,v,keep_mask=torch.ones(2,3,dtype=torch.bool))
    except (TypeError,ValueError,RuntimeError,AssertionError):
        pass
    else:
        raise AssertionError("不可广播的 Mask 必须报告明确错误")
    assert_close(scaled_dot_product_attention(q,k,v,dropout_p=0.75,training=False),torch.ones(1,2,2))


def docs_check(path: Path, required: list[str]):
    assert path.exists(), f"缺少：{path.name}"
    txt=path.read_text(encoding="utf-8")
    assert len(txt.strip()) >= 500, f"{path.name} 内容不足"
    assert "【待完成】" not in txt, f"{path.name} 尚有未填写占位符"
    for term in required:
        assert term.lower() in txt.lower(), f"{path.name} 未体现：{term}"


def test_e1():
    docs_check(ROOT/"Day17_notes.md",["Q", "K", "V", "Softmax", "Padding", "Causal"])


def test_e2():
    docs_check(ROOT/"attention_mask_diagram.md",["Encoder", "Decoder", "Cross-Attention", "True"])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--skip-docs",action="store_true",help="只检查 A–D 代码（最高 90 分）")
    ap.add_argument("--verbose",action="store_true",help="失败时显示完整错误栈")
    args=ap.parse_args()
    cases=[
        ("A1 手算缩放点积和权重",12,test_a1),
        ("A2 Q/K/V 维度和 Cross-Attention",8,test_a2),
        ("B1 Key/Query Padding Mask",10,test_b1),
        ("B2 Causal Mask 与位置偏移",10,test_b2),
        ("B3 Mask 广播及组合",5,test_b3),
        ("C1 改变未来不影响前缀",12,test_c1),
        ("C2 无 Mask 对照实验",8,test_c2),
        ("D1 精确屏蔽与全屏蔽行",10,test_d1),
        ("D2 梯度数值有限",10,test_d2),
        ("D3 输入非法参数/无训练 Dropout",5,test_d3),
    ]
    if not args.skip_docs:
        cases.extend([
            ("E1 理论笔记",5,test_e1),
            ("E2 三种 Attention / Mask 图",5,test_e2),
        ])
    score=0
    for label,points,fn in cases:
        try:
            fn()
        except Exception as exc:
            print(f"[FAIL] {label} (+0/{points}) | {type(exc).__name__}: {exc}")
            if args.verbose:
                traceback.print_exc()
        else:
            score+=points
            print(f"[PASS] {label} (+{points})")
    max_score=90 if args.skip_docs else 100
    print(f"\nDay 17 本次得分：{score}/{max_score}")
    if score < max_score:
        raise SystemExit(1)


if __name__=="__main__":
    main()
