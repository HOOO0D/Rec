"""Day 20｜KV-Cache 与自回归增量推理：学生练习模板。

依赖：Day 17 scaled_attention.py 和 Day 18 multihead_attention.py。
约定：
- 输入 x 已完成 Token/Position 编码，形状 [B,L,d_model]。
- Self-Attention 严格因果；布尔 Mask 中 True=允许读取。
- 每层独立缓存 K/V，形状均为 [B,H,T_cache,d_k]。
- 不缓存历史 Q；只在推理模式使用缓存，缓存不作为可训练参数。
- 测试使用无 Padding 的固定长度玩具序列；复杂变长 Batch/Padding 是以后扩展题。
- 不直接调用官方 DynamicCache / generate；本日目的是独立理解机制。
"""
from __future__ import annotations

from collections.abc import Sequence
import torch
from torch import nn
from multihead_attention import MultiHeadAttention, split_heads, combine_heads
from scaled_attention import make_causal_keep_mask, scaled_dot_product_attention


class LayerKVCache:
    """一层 Self-Attention 的 K/V 缓存；单层示例不是全模型缓存。"""

    def __init__(self) -> None:
        # TODO：设置 self.key=None、self.value=None。
        self.key=None
        self.value = None
        return
        raise NotImplementedError('TODO: LayerKVCache.__init__')

    @property
    def length(self) -> int:
        """已处理的 Token 数；空缓存返回 0。"""
        # TODO：根据 K/V 的序列维（倒数第二维）返回长度。
        if self.key is not None:
            return self.key.shape[-2]
        elif self.value is not None:
            return self.value.shape[-2]
        else:
            return 0
        raise NotImplementedError('TODO: LayerKVCache.length')

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """追加新的 K/V 并返回完整缓存。

        k_new、v_new 形状均为 [B,H,T_new,d_k]，T_new>=1。
        同一层不同增量调用中 B/H/d_k、dtype、device 必须一致。
        注意：不能仅保存最后一块；也不要把历史 Q 放进缓存。
        """
        # TODO：验证维度、非空序列、形状、device、dtype 和浮点类型。
        if k_new.shape != v_new.shape:
            raise ValueError()
        if k_new.ndim != v_new.ndim or k_new.ndim != 4:
            raise ValueError()
        if k_new.shape[-2] <1 or v_new.shape[-2] < 1:
            raise ValueError()
        if k_new.device != v_new.device:
            raise ValueError()
        if k_new.dtype != v_new.dtype:
            raise TypeError()
        if k_new.dtype not in (torch.float16,torch.float32,torch.float64):
            raise TypeError()
        # TODO：第一次 append 直接保存；随后在 dim=-2 使用 torch.cat。
        if self.key is None:
            
            self.key = k_new
            self.value = v_new
        else:
            if k_new.dtype != self.key.dtype:
                raise TypeError("新 K 的 dtype 与历史缓存不一致")
            if v_new.dtype != self.value.dtype:
                raise TypeError("新 V 的 dtype 与历史缓存不一致")
            self.key = torch.cat((self.key,k_new),dim=-2)
            self.value = torch.cat((self.value,v_new),dim=-2)
        # TODO：返回 (self.key, self.value)，不能修改历史缓存的值。
        return (self.key,self.value)
        raise NotImplementedError('TODO: LayerKVCache.append')

    def reset(self) -> None:
        """清除这一层的历史，开始一个新序列。"""
        # TODO：设为 None，不要创建新的可训练参数。
        self.key = None
        self.value = None
        return
        raise NotImplementedError('TODO: LayerKVCache.reset')


def build_cached_causal_keep_mask(
    query_len: int, key_len: int, *, query_start: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """返回 [Lq,Lk] bool Mask，Query 全局位置是 query_start + arange(Lq)。

    例如 Lq=1、Lk=5、query_start=4 时返回 [True]*5；
    而默认的 triu/tril 或非方形 is_causal 不能直接替代这个语义。
    """
    # TODO：复用 Day 17 的 make_causal_keep_mask，并检查长度/起点合法性。
    if query_len<=0 or key_len<=0 or query_start<0:
        raise ValueError()
    mask = make_causal_keep_mask(query_len=query_len,key_len=key_len,query_start=query_start,device=device)
    return mask
    raise NotImplementedError('TODO: build_cached_causal_keep_mask')


class CachedCausalSelfAttention(nn.Module):
    """带 KV-Cache 的因果 MHA；仅实现单层 Attention，不包含 FFN/LayerNorm。"""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        # TODO：建立 self.attn=MultiHeadAttention(d_model,num_heads,dropout_p=0.0)。
        # 注意：必须复用同一套 q_proj/k_proj/v_proj/o_proj 权重，无缓存与有缓存不可各建一套。
        self.attn = MultiHeadAttention(d_model=d_model,num_heads=num_heads,dropout_p=0.0)
        self.d_model = d_model
        self.num_heads = num_heads
        return
        raise NotImplementedError('TODO: CachedCausalSelfAttention.__init__')

    def forward(self, x: torch.Tensor, cache: LayerKVCache | None = None) -> torch.Tensor:
        """无缓存时：整个输入做因果 Self-Attention；有缓存时：只处理新 Token。

        x: [B,L_new,d_model]
        cache=None: 返回完整因果前向 [B,L_new,d_model]
        cache!=None: 更新此层缓存并返回新 Token 输出 [B,L_new,d_model]
        """
        # TODO 1：检查 x 是三维浮点 Tensor，最后一维为 d_model。
        if x.ndim !=3 or x.shape[-1] != self.d_model:
            raise ValueError()
        # TODO 2：cache=None 时：用 Day 17 的因果 Mask，调用 self.attn(x,keep_mask=...)。
        if cache is None:
            L = x.shape[1]
            keep_mask = make_causal_keep_mask(L,device=x.device)
            output = self.attn(x,keep_mask=keep_mask)
            return output
        # TODO 3：cache!=None 时：要求 self.training 为 False，否则报错。
        else:
            if self.training:
                raise ValueError()
        # TODO 4：仅对本次 x 计算 q_proj/k_proj/v_proj，并调用 split_heads。
            q_proj = self.attn.q_proj(x)
            k_proj = self.attn.k_proj(x)
            v_proj = self.attn.v_proj(x)
            q_new = split_heads(q_proj,self.num_heads)
            k_new = split_heads(k_proj,self.num_heads)
            v_new = split_heads(v_proj,self.num_heads)
        # TODO 5：cache.append(k_new,v_new)，得到 k_all、v_all。
            k_all,v_all = cache.append(k_new,v_new)
            L_new = q_new.shape[-2]
        # TODO 6：query_start=cache.length-L_new，构造 [L_new,T_cache] Mask。
            mask = build_cached_causal_keep_mask(query_len = L_new,key_len = cache.length,query_start=cache.length-L_new,device=x.device)
        # TODO 7：使用 Day 17 scaled_dot_product_attention(q_new,k_all,v_all,mask)。
            output = scaled_dot_product_attention(q_new,k_all,v_all,mask)
            output = combine_heads(output)
        # TODO 8：combine_heads 后通过 self.attn.o_proj；返回新 Token 输出。
            output = self.attn.o_proj(output)
            return output
        # 注意：单个 Token 增量 Decode 时 L_new=1，但 Prefill 可以 L_new>1。
        raise NotImplementedError('TODO: CachedCausalSelfAttention.forward')


def make_layer_caches(n_layers: int) -> list[LayerKVCache]:
    """创建 n_layers 个相互独立的缓存实例。"""
    # TODO：检查 n_layers>=1 且是整数；返回互不共享状态的缓存对象。
    if n_layers <1:
        raise ValueError()
    layers = []
    for i in range(n_layers):
        layers.append(LayerKVCache())
    return layers
    raise NotImplementedError('TODO: make_layer_caches')


def compare_full_and_cached(
    layers: Sequence[CachedCausalSelfAttention],
    x: torch.Tensor,
    prefill_len: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """多层 Attention 的无缓存 vs Prefill+增量结果比较。

    此函数是教学版 Attention 堆叠，不包含 Day 19 的 FFN/Post-LN，
    以减少今天的工作范围；Day 21 再组合完整 Transformer。
    输入 x 必须已经包含正确的绝对位置编码，且使用同一批固定参数。

    full: 每层一次处理完整输入；
    cached: 先处理 x[:,:prefill_len]，再逐 Token 处理其余位置；
    返回 full, cached, max_abs_error。
    """
    # TODO 1：检查 x [B,T,d]、0<prefill_len<=T、layers 非空且均在 eval()。
    if x.ndim !=3:
        raise ValueError()
    T = x.shape[-2]
    if prefill_len<=0 or prefill_len> T:
        raise ValueError()
    if len(layers) ==0:
        raise ValueError()
    for layer in layers:
        if layer.training:
            raise ValueError()
    # TODO 2：依次完整前向 through 各层，记录 full（不传缓存）。
    with torch.no_grad():
        output_full = None
        for i in range(len(layers)):
            if i == 0:
                output_full = layers[i](x)
            else:
                output_full = layers[i](output_full)
    # TODO 3：每层创建独立缓存，先做 Prefill；后续按 Token 增量 Decode。
    with torch.no_grad():
        cache_layers = make_layer_caches(n_layers=len(layers))
        outputs = []
        h = x[:,:prefill_len,:]
        for layer,cache in zip(layers,cache_layers):
            h = layer(h,cache=cache)
        outputs.append(h)
        for t in range(prefill_len,T):
            h = x[:,t:t+1,:]
            for layer,cache in zip(layers,cache_layers):
                h = layer(h,cache=cache)
            outputs.append(h)
    # TODO 4：拼接 [B,prefill_len,d] + 剩余各 [B,1,d]。
    cached = torch.cat(outputs,dim=1)
    # TODO 5：返回 full、cached、max_abs_error（浮点数）。
    return output_full,cached,(output_full - cached).abs().max().item()
    # 提示：优先在 torch.no_grad() 下运行，比较结果而不是训练。
    raise NotImplementedError('TODO: compare_full_and_cached')
