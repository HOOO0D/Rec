"""Day 18 自动验收：A–E 代码 90 分，F 阅读作业 10 分。

要先完成 Day 17 的 scaled_attention.py，放在当前目录。
本测试可使用官方 nn.MultiheadAttention 作为独立数值参照；学生实现不可直接包装该模块。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import traceback
import torch
from torch import nn
from multihead_attention import MultiHeadAttention, split_heads, combine_heads, expand_keep_mask

ROOT = Path(__file__).resolve().parent

def assert_close(got, expected, atol=1e-5, rtol=1e-5):
    assert tuple(got.shape) == tuple(expected.shape), f'形状不一致：{tuple(got.shape)} != {tuple(expected.shape)}'
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)

def to_official(model):
    """映射学生模型到 PyTorch 官方 MHA，用于独立对照（合法非全屏蔽输入）。"""
    ref = nn.MultiheadAttention(model.d_model,model.num_heads, dropout=0.,bias=(model.q_proj.bias is not None),batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([model.q_proj.weight,model.k_proj.weight,model.v_proj.weight],dim=0))
        if model.q_proj.bias is not None:
            ref.in_proj_bias.copy_(torch.cat([model.q_proj.bias,model.k_proj.bias,model.v_proj.bias]))
        ref.out_proj.weight.copy_(model.o_proj.weight)
        if model.o_proj.bias is not None: ref.out_proj.bias.copy_(model.o_proj.bias)
    return ref.eval()

def test_a1():
    x=torch.arange(2*3*8,dtype=torch.float32).reshape(2,3,8)
    s=split_heads(x,2)
    assert s.shape==(2,2,3,4)
    assert_close(s[0,0,0],torch.tensor([0.,1.,2.,3.]))
    assert_close(s[0,1,0],torch.tensor([4.,5.,6.,7.]))
    assert_close(combine_heads(s),x)

def test_a2():
    for bad_h in [0,3,-2]:
        try: split_heads(torch.zeros(2,3,8),bad_h)
        except (ValueError,TypeError): pass
        else: raise AssertionError(f'num_heads={bad_h} 应当报错')
    try: combine_heads(torch.zeros(2,3,8))
    except (ValueError,TypeError): pass
    else: raise AssertionError('combine_heads 输入必须是四维')

def test_b1():
    torch.manual_seed(18)
    m=MultiHeadAttention(8,2,bias=True).eval()
    assert m.head_dim == 4
    assert sum(p.numel() for p in m.parameters())==4*(8*8+8)
    x=torch.randn(2,4,8)
    got,w=m(x,return_weights=True)
    ref=to_official(m)
    expected,wref=ref(x,x,x,need_weights=True,average_attn_weights=False)
    assert_close(got,expected)
    assert_close(w,wref)
    assert w.shape==(2,2,4,4)

def test_b2():
    torch.manual_seed(12)
    m=MultiHeadAttention(8,2,bias=False).eval()
    assert sum(p.numel() for p in m.parameters())==4*8*8
    x=torch.randn(1,3,8)
    y=m(x)
    with torch.no_grad(): m.o_proj.weight.zero_()
    y0=m(x)
    assert y0.abs().max().item()<1e-7, '输出必须经过 W_O；设为零时结果应变为零'
    assert not torch.allclose(y,y0), '输出投影检查需要有变化'

def test_c1():
    torch.manual_seed(7)
    m=MultiHeadAttention(8,2).eval()
    x=torch.randn(2,4,8)
    # Day17 的 Mask：True=可以访问；官方 MHA 布尔 mask 的 True=屏蔽，测试中取反。
    causal=torch.tril(torch.ones(4,4,dtype=torch.bool))
    out,w=m(x,keep_mask=causal,return_weights=True)
    official,woff=to_official(m)(x,x,x,attn_mask=~causal,need_weights=True,average_attn_weights=False)
    assert_close(out,official)
    assert_close(w,woff)
    assert_close(w.masked_select((~causal)[None,None]),torch.zeros_like(w.masked_select((~causal)[None,None])))

def test_c2():
    torch.manual_seed(123)
    m=MultiHeadAttention(8,2).eval()
    x=torch.randn(2,4,8)
    changed=x.clone(); changed[:,3]+=torch.tensor([100.,-50.,20.,-7.,2.,30.,5.,-3.])
    mask=torch.tril(torch.ones(4,4,dtype=torch.bool))
    assert_close(m(x,keep_mask=mask)[:,:3],m(changed,keep_mask=mask)[:,:3])
    assert (m(x)[:,:3]-m(changed)[:,:3]).abs().max().item()>1e-4
    # 对所有 Head 都没有可用 Key 的 Query，其最终输出约定为零（即使有输出 bias）。
    mask2=mask.clone(); mask2[2,:]=False
    out,w=m(x,keep_mask=mask2,return_weights=True)
    assert torch.isfinite(out).all() and torch.isfinite(w).all()
    assert_close(out[:,2],torch.zeros_like(out[:,2]))
    assert_close(w[:,:,2],torch.zeros_like(w[:,:,2]))

def test_d1():
    torch.manual_seed(16)
    m=MultiHeadAttention(8,2).eval()
    q=torch.randn(2,3,8); k=torch.randn(2,5,8); v=torch.randn(2,5,8)
    mask=torch.ones(2,3,5,dtype=torch.bool); mask[:,:,-1]=False
    out,w=m(q,k,v,keep_mask=mask,return_weights=True)
    ref=to_official(m)
    # 按 batch 的动态 mask，官方接口使用 [B*H,Lq,Lk]
    disallow=(~mask).unsqueeze(1).expand(2,2,3,5).reshape(4,3,5)
    expected,wr=ref(q,k,v,attn_mask=disallow,need_weights=True,average_attn_weights=False)
    assert out.shape==(2,3,8) and w.shape==(2,2,3,5)
    assert_close(out,expected); assert_close(w,wr)

def test_d2():
    m=MultiHeadAttention(8,2).eval()
    q=torch.randn(2,3,8); k=torch.randn(2,5,8); v=torch.randn(2,5,8)
    m1=torch.ones(2,1,5,dtype=torch.bool)
    m2=torch.ones(2,3,1,dtype=torch.bool)
    m1[:,:,-1]=False
    m2[:,2,:]=False
    expanded=expand_keep_mask(m1 & m2,batch_size=2,num_heads=2,query_len=3,key_len=5,device=q.device)
    assert expanded.shape==(2,2,3,5)
    assert expanded.dtype==torch.bool
    out,w=m(q,k,v,keep_mask=m1&m2,return_weights=True)
    assert_close(w[:,:,2],torch.zeros_like(w[:,:,2]))
    assert_close(out[:,2],torch.zeros_like(out[:,2]))
    assert_close(w[:,:,:,-1],torch.zeros_like(w[:,:,:,-1]))

def test_e1():
    torch.manual_seed(218)
    m=MultiHeadAttention(8,2).train()
    q=torch.randn(2,3,8,requires_grad=True)
    k=torch.randn(2,4,8,requires_grad=True)
    v=torch.randn(2,4,8,requires_grad=True)
    out=m(q,k,v)
    loss=out.square().mean(); loss.backward()
    for name in ['q_proj','k_proj','v_proj','o_proj']:
        p=getattr(m,name).weight
        assert p.grad is not None, f'{name} 无梯度'
        assert torch.isfinite(p.grad).all(), f'{name} 梯度非有限'
        assert p.grad.abs().sum().item()>0, f'{name} 梯度全零'
    for tensor in (q,k,v):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()

def test_e2():
    for d,h in [(8,0),(8,3),(0,2),(-4,2)]:
        try: MultiHeadAttention(d,h)
        except (ValueError,TypeError): pass
        else: raise AssertionError(f'd_model={d},h={h} 应报错')
    m=MultiHeadAttention(8,2)
    x=torch.randn(1,4,8)
    for fn in [lambda:m(x,torch.randn(1,4,8)),lambda:m(torch.randn(2,4,8),torch.randn(1,4,8),torch.randn(1,4,8)),lambda:m(x,keep_mask=torch.ones(4,4)),lambda:m(x,keep_mask=torch.ones(3,4,dtype=torch.bool))]:
        try: fn()
        except (ValueError,TypeError,RuntimeError): pass
        else: raise AssertionError('非法输入未抛异常')

def test_f1():
    for filename in ['Day18_notes.md','attention_heads_diagram.md']:
        content=(ROOT/filename).read_text(encoding='utf-8')
        assert len(content.strip())>650, f'{filename} 内容不足'
        assert 'TODO' not in content and '（填写' not in content, f'{filename} 存在未完成提示'
    notes=(ROOT/'Day18_notes.md').read_text(encoding='utf-8')
    assert 'Self-Attention' in notes and 'Cross-Attention' in notes
    assert 'W_Q' in notes and 'W_O' in notes

CASES=[
 ('A1 拆头/拼头数值',8,test_a1),('A2 不合法头数',7,test_a2),
 ('B1 与 PyTorch 官方逐头输出一致',15,test_b1),('B2 W_O 和无偏置参数审计',10,test_b2),
 ('C1 Causal Mask/官方结果核对',10,test_c1),('C2 无未来泄漏/全屏蔽',10,test_c2),
 ('D1 Cross-Attention 不等长输入',10,test_d1),('D2 Padding/Query 行/头维广播',5,test_d2),
 ('E1 四投影梯度与输入梯度',10,test_e1),('E2 异常和非法输入',5,test_e2),
 ('F1 理论笔记/头结构图',10,test_f1)
]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-docs',action='store_true',help='暂不检查理论作业：最高 90 分')
    parser.add_argument('--verbose',action='store_true')
    args=parser.parse_args()
    total=0; limit=0; failed=[]
    for name,score,fn in CASES:
        if args.skip_docs and name.startswith('F1'):continue
        limit+=score
        try:
            fn(); total+=score; print(f'[PASS] {name} (+{score})')
        except Exception as exc:
            failed.append(name)
            print(f'[FAIL] {name} (0/{score}): {type(exc).__name__}: {exc}')
            if args.verbose: traceback.print_exc()
    print(f'\nDay 18 得分：{total}/{limit}；未通过 {len(failed)} 项')
    if failed: raise SystemExit(1)

if __name__=='__main__': main()
