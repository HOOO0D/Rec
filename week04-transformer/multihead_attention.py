"""Day 18：从零实现 Multi-Head Attention 的学生模板。

接口约定：
- 输入使用 batch_first，形状 [B,L,d_model]。
- Self-Attention：只传 query，自动令 key=value=query。
- Cross-Attention：必须同时传 query、key、value；允许 Lq != Lk。
- 布尔 keep_mask 中 True=允许访问，False=屏蔽；延续 Day 17。
- keep_mask 允许 [Lq,Lk]、[B,Lq,Lk]、[B,1,Lk]、[B,Lq,1]、[B,1,Lq,Lk]、[B,H,Lq,Lk]。
- return_weights=True 时返回 (output, weights)，weights:[B,H,Lq,Lk]；否则仅返回 output。
- 对所有 Head 都没有合法 Key 的 Query 行，最终输出约定为零，即使输出投影含 bias。
- 可以复用自己在 Day 17 完成的 scaled_dot_product_attention；不要直接调用 nn.MultiheadAttention 作为实现。
"""
from __future__ import annotations
import torch
from torch import nn
from scaled_attention import scaled_dot_product_attention


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[B,L,d_model] -> [B,H,L,d_k]，其中 d_model=H*d_k。"""
    # TODO：检查 x 的形状、num_heads 为正整数、d_model 可整除。
    if x.ndim != 3:
        raise ValueError()
    if num_heads <=0:
        raise ValueError()
    if x.shape[-1] % num_heads != 0:
        raise ValueError()
    # TODO：先 reshape 成 [B,L,H,d_k]，再交换第 1、2 维。
    B,L,d_model = x.shape
    x = x.reshape(B,L,num_heads,d_model//num_heads).transpose(1,2)
    return x
    raise NotImplementedError('TODO: split_heads')


def combine_heads(x: torch.Tensor) -> torch.Tensor:
    """[B,H,L,d_v] -> [B,L,H*d_v]，与 split_heads 顺序互逆。"""
    # TODO：检查输入维度；通过 transpose + contiguous/reshape 拼头。
    if x.ndim != 4:
        raise ValueError()
    B,H,L,d_v = x.shape
    x = x.transpose(1,2).reshape(B,L,H*d_v)
    # TODO：不要直接 view [B,L,H*d_v]，这样会打乱维度顺序。
    return x
    raise NotImplementedError('TODO: combine_heads')


def expand_keep_mask(
    keep_mask: torch.Tensor | None,
    *, batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    """标准化为 [B,H,Lq,Lk] 的 bool 掩码；None 则返回 None。

    支持 Day17 的 Key Padding [B,1,Lk]、Query Padding [B,Lq,1]，
    以及组合后的 [B,Lq,Lk]；二维矩阵视为跨 Batch/Head 共享。
    """
    # TODO：要求布尔掩码；形状仅允许 2/3/4 维；须在相同 device。
    if keep_mask is None:
        return None
    if keep_mask.dtype != torch.bool:
        raise TypeError()
    if keep_mask.ndim not in (2,3,4):
        raise ValueError()
    if keep_mask.device!=device:
        raise ValueError()
    # TODO：二维加两个前缀维；三维加 Head 维；四维保持不变。
    if keep_mask.ndim == 2:
        keep_mask = keep_mask.unsqueeze(0).unsqueeze(0)
    elif keep_mask.ndim == 3:
        keep_mask = keep_mask.unsqueeze(1)
    # TODO：验证可以广播到目标 [B,H,Lq,Lk] 并显式 expand。
    if keep_mask is not None:
        try:
            keep_mask = keep_mask.expand(batch_size, num_heads, query_len, key_len)
        except RuntimeError as exc:
            raise ValueError(
            f"keep_mask 的形状 {tuple(keep_mask.shape)} "
            f"不能广播到 {(batch_size, num_heads, query_len, key_len)}"
            ) from exc
        return keep_mask
    raise NotImplementedError('TODO: expand_keep_mask')


class MultiHeadAttention(nn.Module):
    """教学版 MHA：Q/K/V 各自投影，逐头 SDPA，拼接后 W_O 投影。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # TODO：验证 d_model、num_heads 是正整数且整除；0 <= dropout_p < 1。
        if d_model<=0 or num_heads<=0:
            raise ValueError()
        if not (0<=dropout_p<1):
            raise ValueError()
        if d_model % num_heads!= 0:
            raise ValueError()
        # TODO：保存 d_model、num_heads、head_dim、dropout_p。
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        self.dropout_p = dropout_p

        # TODO：建立 q_proj/k_proj/v_proj/o_proj 四个 nn.Linear(d_model,d_model,bias=bias)。
        # 注意：原论文中各 Head 可写成独立 W_i，本实现把它们打包在大矩阵中实现等价投影。
        self.q_proj = nn.Linear(self.d_model,self.d_model,bias=bias)
        self.k_proj = nn.Linear(self.d_model,self.d_model,bias=bias)
        self.v_proj = nn.Linear(self.d_model,self.d_model,bias=bias)
        self.o_proj = nn.Linear(self.d_model,self.d_model,bias=bias)
        return 
        raise NotImplementedError('TODO: MultiHeadAttention.__init__')

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        keep_mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """自注意力或交叉注意力。

        query:[B,Lq,d_model]; key/value:[B,Lk,d_model]。
        Return: output:[B,Lq,d_model]；权重:[B,H,Lq,Lk]。
        """
        # TODO 1：key/value 均未传时按 self-attention；只传其一时报 ValueError。
        if key is None or value is None:
            if not (key is None and value is None):
                raise ValueError()
            else:
                key = value = query
        # TODO 2：验证三者形状、Batch/序列长度、设备、浮点 dtype。
        if query.ndim != 3 or key.ndim !=3 or value.ndim!=3:
            raise ValueError()
        if not (query.shape[0]==key.shape[0]==value.shape[0]):
            raise ValueError()
        if not (query.device==value.device==key.device):
            raise TypeError()
        if not (query.dtype==key.dtype==value.dtype):
            raise TypeError()
        if query.dtype not in (torch.float64,torch.float32,torch.float16):
            raise TypeError()
        # TODO 3：分别投影 Q/K/V，然后 split_heads。
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        Q = split_heads(Q,self.num_heads)
        K = split_heads(K,self.num_heads)
        V = split_heads(V,self.num_heads)
        # TODO 4：调用 expand_keep_mask，得到 [B,H,Lq,Lk]。
        B = Q.shape[0]
        H = Q.shape[1]
        Lq = Q.shape[2]
        Lk = K.shape[2]
        mask = expand_keep_mask(keep_mask,batch_size=B,query_len=Lq,key_len=Lk,device=Q.device,num_heads=self.num_heads)

        # TODO 5：复用 Day17 SDPA；training=self.training，dropout_p=self.dropout_p。
        if return_weights:
            [output,weights] = scaled_dot_product_attention(Q,K,V,mask,dropout_p=self.dropout_p,return_weights=return_weights,training=self.training)
        else:
            output = scaled_dot_product_attention(Q,K,V,mask,dropout_p=self.dropout_p,return_weights=return_weights,training=self.training)
        # TODO 6：combine_heads，再通过 o_proj 得到最终输出。
        output = combine_heads(output)
        output = self.o_proj(output)
        
        # TODO 7：如果某 Query 在所有 Head 中均无可访问 Key，输出整行归零；返回逐头权重。
        if mask is not None:
            head_has_key = mask.any(dim=-1)          # [B,H,Lq]
            query_has_key = head_has_key.any(dim=1)  # [B,Lq]

            output = output.masked_fill(
                ~query_has_key.unsqueeze(-1), 0.0
            )

        if return_weights:
            return output, weights
        return output
        raise NotImplementedError('TODO: MultiHeadAttention.forward')
