"""Day 20 KV-Cache 自动验收：A–E=90 分、F=10 分。

在已完成 Day16/17/18 代码的同一目录运行；不要把测试脚本当作实现参考答案。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import traceback
import torch
from torch import nn
from kv_cache import (
    LayerKVCache, build_cached_causal_keep_mask,
    CachedCausalSelfAttention, make_layer_caches, compare_full_and_cached
)

ROOT=Path(__file__).resolve().parent


def close(a:torch.Tensor,b:torch.Tensor,atol=1e-5):
    torch.testing.assert_close(a,b,atol=atol,rtol=1e-5)


def raises(fn, exceptions=(ValueError,TypeError,RuntimeError,AssertionError)):
    try: fn()
    except exceptions: return
    raise AssertionError('预期非法参数报错，但没有抛异常')


def test_a1():
    c=LayerKVCache()
    assert c.length==0 and c.key is None and c.value is None
    k1=torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)
    v1=100+k1
    k,v=c.append(k1,v1)
    assert c.length==3 and k.shape==v.shape==(1,2,3,4)
    close(k,k1);close(v,v1)
    k2=torch.randn(1,2,1,4);v2=torch.randn(1,2,1,4)
    k,v=c.append(k2,v2)
    assert c.length==4 and k.shape==(1,2,4,4)
    close(k[:,:,:3],k1);close(v[:,:,:3],v1)
    close(k[:,:,3:4],k2);close(v[:,:,3:4],v2)
    assert c.key is not None and c.value is not None
    assert all(not isinstance(getattr(c,x), nn.Parameter) for x in ('key','value'))
    c.reset()
    assert c.length==0 and c.key is None and c.value is None


def test_a2():
    c=LayerKVCache()
    raises(lambda:c.append(torch.randn(2,3,4),torch.randn(2,3,4)))
    raises(lambda:c.append(torch.randn(1,2,0,4),torch.randn(1,2,0,4)))
    raises(lambda:c.append(torch.randn(1,2,2,4),torch.randn(1,2,3,4)))
    c.append(torch.randn(1,2,2,4),torch.randn(1,2,2,4))
    raises(lambda:c.append(torch.randn(1,3,1,4),torch.randn(1,3,1,4)))
    raises(lambda:c.append(torch.randn(1,2,1,5),torch.randn(1,2,1,5)))
    raises(lambda:c.append(torch.randn(1,2,1,4,dtype=torch.float64),torch.randn(1,2,1,4,dtype=torch.float64)))
    assert c.length==2,'非法 append 不应改变原缓存'


def test_b1():
    m=build_cached_causal_keep_mask(1,4,query_start=3)
    assert m.dtype==torch.bool and m.shape==(1,4)
    close(m,torch.ones(1,4,dtype=torch.bool))
    m=build_cached_causal_keep_mask(2,5,query_start=3)
    close(m,torch.tensor([[1,1,1,1,0],[1,1,1,1,1]],dtype=torch.bool))
    m=build_cached_causal_keep_mask(3,3,query_start=0)
    close(m,torch.ones(3,3,dtype=torch.bool).tril())
    assert m.device.type=='cpu'


def test_b2():
    for args in [(0,4,0),(1,0,0),(1,4,-1)]:
        raises(lambda a=args:build_cached_causal_keep_mask(a[0],a[1],query_start=a[2]))
    mask=build_cached_causal_keep_mask(1,5,query_start=4)
    # 错误的左上对齐只允许访问第一个 Key，这里应允许全部五个。
    assert int(mask.sum())==5
    c=make_layer_caches(3)
    assert len(c)==3 and len({id(x) for x in c})==3
    raises(lambda:make_layer_caches(0))
    raises(lambda:make_layer_caches(-1))


def test_c1():
    torch.manual_seed(2026)
    layer=CachedCausalSelfAttention(8,2).eval()
    assert sum(p.numel() for p in layer.parameters())==288
    x=torch.randn(2,6,8)
    with torch.no_grad():
        full=layer(x)
        cache=LayerKVCache()
        parts=[layer(x[:,:3],cache)]
        assert cache.length==3
        for i in range(3,6):
            parts.append(layer(x[:,i:i+1],cache))
            assert cache.length==i+1
        inc=torch.cat(parts,dim=1)
        close(inc,full)
        assert max((full-inc).abs().reshape(-1)).item()<=1e-5
        changed=x.clone(); changed[:,5,:]+=25
        close(layer(changed)[:,:5],full[:,:5])
    assert full.shape==(2,6,8)


def test_c2():
    torch.manual_seed(41)
    layer=CachedCausalSelfAttention(8,2).eval()
    x=torch.randn(1,5,8)
    lengths=[]
    hook=layer.attn.k_proj.register_forward_hook(lambda _,args,out:lengths.append(args[0].size(1)))
    with torch.no_grad():
        c=LayerKVCache()
        layer(x[:,:3],c)
        layer(x[:,3:4],c)
        layer(x[:,4:5],c)
    hook.remove()
    assert lengths==[3,1,1],f'增量阶段应只投影新 Token，实际 K 投影长度 {lengths}'
    layer.train()
    raises(lambda:layer(x[:,:1],LayerKVCache()),(ValueError,TypeError,RuntimeError,AssertionError))
    layer.eval()
    raises(lambda:CachedCausalSelfAttention(7,2))


def test_d1():
    torch.manual_seed(42)
    layers=[CachedCausalSelfAttention(8,2).eval() for _ in range(2)]
    x=torch.randn(2,7,8)
    for pref in (1,3,7):
        full,inc,err=compare_full_and_cached(layers,x,prefill_len=pref)
        assert full.shape==inc.shape==(2,7,8)
        close(full,inc)
        assert isinstance(err,float) and err<=1e-5,(pref,err)
    raises(lambda:compare_full_and_cached(layers,x,prefill_len=0))
    raises(lambda:compare_full_and_cached(layers,x,prefill_len=8))


def test_d2():
    layers=[CachedCausalSelfAttention(8,2).eval() for _ in range(3)]
    caches=make_layer_caches(len(layers))
    x=torch.randn(1,3,8)
    with torch.no_grad():
        z=x
        for layer,cache in zip(layers,caches):
            z=layer(z,cache)
        assert [c.length for c in caches]==[3,3,3]
        z=torch.randn(1,1,8)
        for layer,cache in zip(layers,caches):z=layer(z,cache)
        assert [c.length for c in caches]==[4,4,4]
    caches[0].reset()
    assert [c.length for c in caches]==[0,4,4], '每层缓存必须独立'


def test_e1():
    from embeddings_position import TransformerInputEmbedding
    torch.manual_seed(55)
    embed=TransformerInputEmbedding(32,8,max_len=32).eval()
    tokens=torch.tensor([[3,6,4,9,2]],dtype=torch.long)
    with torch.no_grad():
        full=embed(tokens,start_pos=0)
        pieces=[embed(tokens[:,:3],start_pos=0)]
        for t in (3,4):pieces.append(embed(tokens[:,t:t+1],start_pos=t))
        joined=torch.cat(pieces,dim=1)
        close(joined,full)
        wrong=embed(tokens[:,3:4],start_pos=0)
        assert not torch.allclose(wrong,full[:,3:4]), '错误的绝对位置必须能被发现'
    assert full.shape==(1,5,8)


def test_e2():
    torch.manual_seed(12)
    x=torch.randn(1,6,8)
    layer=CachedCausalSelfAttention(8,2).eval()
    with torch.no_grad():
        full,inc,err=compare_full_and_cached([layer],x,prefill_len=2)
    assert err<=1e-5
    # 6 个新 Token 无缓存逐前缀全部投影：1+...+6=21 个位置；有缓存各位置只投影一次=6。
    no_cache_tokens=sum(range(1,7))
    cached_tokens=6
    assert no_cache_tokens==21 and cached_tokens==6
    # 这只统计 Q/K/V 投影的 Token 次数，不代表真实运行时间能严格按此比值加速。


def test_f():
    path=ROOT/'Day20_notes.md'
    assert path.exists(),'缺少 Day20_notes.md'
    s=path.read_text(encoding='utf-8')
    assert 'TODO' not in s, '理论记录中仍有 TODO 未填写'
    assert len(s)>=1600,'记录过短，请按模板完成原理、形状、数值和效率分析'
    for term in ['Prefill','Decode','Query','Mask','缓存','显存']:
        assert term.lower() in s.lower(),f'记录遗漏关键概念：{term}'


TESTS=[
    ('A1 缓存增长与清空',12,test_a1),('A2 形状与非法输入',8,test_a2),
    ('B1 绝对位置因果 Mask',10,test_b1),('B2 不等长与多层缓存',5,test_b2),
    ('C1 单层 Prefill/Decode 等价',15,test_c1),('C2 只投影新 Token / 推理约束',10,test_c2),
    ('D1 多层等价',12,test_d1),('D2 每层缓存独立',8,test_d2),
    ('E1 位置编码偏移',6,test_e1),('E2 计算量对照',4,test_e2),
    ('F 理论笔记',10,test_f),
]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--skip-docs',action='store_true')
    ap.add_argument('--verbose',action='store_true')
    args=ap.parse_args()
    total=0; possible=0
    for name,points,func in TESTS:
        if args.skip_docs and name.startswith('F '):continue
        possible+=points
        try:
            func()
        except Exception as e:
            print(f'[FAIL] {name} (0/{points}): {type(e).__name__}: {e}')
            if args.verbose:traceback.print_exc()
        else:
            total+=points
            print(f'[PASS] {name} (+{points})')
    print(f'\nDay20 分数：{total}/{possible}')
    if total!=possible:raise SystemExit(1)

if __name__=='__main__':main()
