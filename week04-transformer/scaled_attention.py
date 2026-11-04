"""Day 17：缩放点积注意力与 Padding/Causal Mask（学生练习模板）。

约定：
1. 本日只实现已经给定 Q、K、V 的注意力；Day 18 再实现 W_Q/W_K/W_V/W_O。
2. Q:[B,Lq,dk]、K:[B,Lk,dk]、V:[B,Lk,dv]，可选支持带 Head 维的批次。
3. 所有布尔 keep_mask：True=允许读取，False=禁止读取。
4. 全屏蔽 Query 行：注意力权重全 0，输出全 0，不能出现 NaN/Inf。
5. return_weights=True 时返回 (output, weights)；False 时只返回 output。
6. 单头注意力默认不使用 Dropout；保留 training/dropout_p 接口供后续扩展。

请独立实现 TODO，不要直接调用 F.scaled_dot_product_attention 来替代本次的核心实现。
"""
from __future__ import annotations

import torch


def make_padding_keep_mask(
    token_ids: torch.Tensor,
    padding_idx: int = 0,
    *,
    for_query: bool = False,
) -> torch.Tensor:
    """根据 [B,L] token ID 构造可广播的布尔 Padding 掩码。

    for_query=False：返回 [B,1,L]，屏蔽无效 Key 列。
    for_query=True ：返回 [B,L,1]，屏蔽无效 Query 行。
    """
    # TODO：检查 token_ids 是二维整数张量，并检查 padding_idx。
    if token_ids.ndim !=2 :
        raise ValueError()
    if padding_idx <0 :
        raise ValueError()
    # TODO：有效 token 为 True，Padding 为 False。
    mask = (token_ids!=padding_idx)
    # TODO：按 for_query 决定在第 1 维还是最后一维增加维度。
    if for_query:
        return mask.unsqueeze(-1)
    else :
        return mask.unsqueeze(1)
    raise NotImplementedError("TODO: make_padding_keep_mask")


def make_causal_keep_mask(
    query_len: int,
    key_len: int | None = None,
    *,
    query_start: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """返回 [Lq,Lk] 下三角式掩码，True=该 Query 允许访问该 Key。

    第 r 个 Query 的全局位置是 query_start+r。
    key_len 默认等于 query_len；query_start 默认 0。
    例如 Lq=1,Lk=4,query_start=3 时，该 Query 可访问四个 Key。
    """
    # TODO：检查长度是合法正整数，query_start 是非负整数。
    if key_len is None:
        key_len = query_len
    if query_len <=0 or query_start<0:
        raise ValueError()
    if not isinstance(query_len,int) or not isinstance(query_start,int):
        raise TypeError()
    # TODO：计算 Query 与 Key 的位置序号，允许条件 key_pos <= query_pos。
    query_pos = (torch.arange(query_len,device=device)+query_start).unsqueeze(1)
    key_pos = (torch.arange(key_len,device=device)).unsqueeze(0)
    # TODO：返回 bool 张量，且放置到要求的 device。
    return key_pos<=query_pos
    raise NotImplementedError("TODO: make_causal_keep_mask")


def combine_keep_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """对若干 bool 掩码取逻辑与；参数允许为 None，支持 PyTorch 广播。

    如：Key Padding[B,1,L] & Query Padding[B,L,1] & Causal[L,L]
    → [B,L,L]。全部传 None 时返回 None。
    """
    # TODO：过滤 None，校验剩余掩码 dtype 为 torch.bool。
    if masks is None:
        return None
    mask_total = None
    for mask in masks:
        if mask is None:
            continue
        if mask.dtype != torch.bool:
            raise TypeError()
        if mask_total is None:
            mask_total = mask
        else:
            mask_total = mask_total & mask
    # TODO：将所有掩码按位与，依靠广播合并。
    return mask_total
    raise NotImplementedError("TODO: combine_keep_masks")


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keep_mask: torch.Tensor | None = None,
    *,
    dropout_p: float = 0.0,
    training: bool = False,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """单头缩放点积注意力；可选支持 [B,H,L,D] 的批次前缀。

    Args:
        query: [...,Lq,dk]
        key: [...,Lk,dk]
        value: [...,Lk,dv]
        keep_mask: 布尔张量，可广播到 [...,Lq,Lk]；True=允许访问。
        dropout_p: 在 training=True 时，对注意力权重施加 Dropout。
        return_weights: 返回未 Dropout 的权重，便于验收。
    Returns:
        output: [...,Lq,dv]
        weights: [...,Lq,Lk] （若 return_weights=True）
    """
    # TODO 1：检查 Q/K/V 的形状、dk 和 Lk 是否匹配、dtype 和 device。
    if query.ndim <2 or key.ndim<2 or value.ndim<2:
        raise ValueError()
    # Q 和 K 的向量维度必须一致，才能计算点积
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q 和 K 的 dk 必须一致")

    # 每个 Key 必须对应一个 Value
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("K 和 V 的 Lk 必须一致")

    dk = query.shape[-1]

    if dk <= 0:
        raise ValueError("dk 必须大于 0")
    if not (query.dtype == key.dtype == value.dtype):
        raise TypeError()
    if not (query.device == key.device == value.device):
        raise ValueError()
    if query.dtype not in (torch.float16,torch.float32,torch.float64):
        raise TypeError()
    # TODO 2：scores = (Q @ K.transpose(-2,-1)) / sqrt(dk)。
    scores = (query@key.transpose(-2,-1))/ (dk**0.5)
    # TODO 3：把布尔 keep_mask 广播到 scores.shape，屏蔽位置设为 -inf。
    if keep_mask is not None:
        scores = scores.masked_fill(~keep_mask,-torch.inf)
    # TODO 4：特别处理没有任何有效 Key 的 Query 行：防止 softmax(-inf,...,-inf) 产生 NaN。
        # keep_mask 已广播到 scores.shape
        row_has_key = keep_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~row_has_key, 0.0)

    # TODO 5：沿 Key 维度 softmax；被屏蔽位置权重必须严格为 0。
    weights = torch.softmax(scores,dim=-1)
    if keep_mask is not None:
        weights = weights.masked_fill(~row_has_key,0.0)
    # TODO 6：如果需要 Dropout，训练时对权重应用；eval 或 training=False 不启用。
    weights_drop = torch.nn.functional.dropout(weights,p=dropout_p,training=training)
    # TODO 7：output = weights @ V，按 return_weights 决定输出接口。
    output = weights_drop @ value
    if return_weights:
        return output,weights
    return output
    # 说明：测试不接受将该函数直接替换为 PyTorch 的现成 SDPA；需掌握原始计算。
    raise NotImplementedError("TODO: scaled_dot_product_attention")
