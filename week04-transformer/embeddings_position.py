"""Day 16：Transformer 输入表示（学生练习模板）。

实现说明：
1. 约定 token_ids 为 LongTensor，形状 [B,L]；0 是 Padding ID。
2. 用 nn.Embedding 完成查表并乘 sqrt(d_model)。
3. 位置编码使用原论文的正弦/余弦公式，必须通过 register_buffer 保存。
4. 有效位置约定为 True；在本练习中将 Padding 位置的最终输入向量置零。
5. 本文件只构建输入，不负责 Day 17 的 Attention、Causal Mask。

所有 TODO 请独立完成；运行 Test_Day16.py 验收。
"""

from __future__ import annotations

import math
import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Token ID → 可训练的 token 向量，并做 sqrt(d_model) 缩放。"""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        # TODO：检查 vocab_size、d_model、padding_idx 的合法性。
        if vocab_size <=0 or d_model<=0 or padding_idx<0 or padding_idx>=vocab_size:
            raise ValueError()
        # TODO：创建 self.embedding = nn.Embedding(..., padding_idx=...)。
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model,padding_idx=padding_idx)
        # TODO：记录维度缩放系数；padding_idx=0 的权重默认初始化为 0。
        self.scale = math.sqrt(d_model)
        return 
        raise NotImplementedError("TODO: TokenEmbedding.__init__")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # TODO：验证输入形状为 [B,L]，数据类型为 torch.long。
        if token_ids.dtype != torch.long:
            raise ValueError()
        if token_ids.ndim != 2:
            raise ValueError()
        # TODO：查表并乘 sqrt(d_model)，返回 [B,L,d_model]。
        token = self.embedding(token_ids)*self.scale
        return token
        raise NotImplementedError("TODO: TokenEmbedding.forward")


class SinusoidalPositionalEncoding(nn.Module):
    """固定正弦/余弦位置编码；pe 应为 [max_len,d_model] 的 Buffer。"""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        # TODO：检查 d_model 为正偶数、max_len 为正数。
        if d_model <=0 or d_model%2 !=0:
            raise ValueError()
        if max_len<=0:
            raise ValueError()
        # TODO：用 torch.arange 构造 pos 和偶数维索引。
        pos = torch.arange(max_len).unsqueeze(1)
        even = torch.arange(0,d_model,2)
        # TODO：偶数维填充 sin(pos / 10000^(2i/d_model))。
        denominator = 10000.0**(even/d_model)
        angel = pos/denominator
        # TODO：奇数维填充 cos(pos / 10000^(2i/d_model))。
        pe = torch.zeros(max_len,d_model)
        pe[:,0::2] = torch.sin(angel)
        pe[:,1::2] = torch.cos(angel)
        # 三段式切片 起点：终点：步长
        # TODO：self.register_buffer("pe", pe)，不要把 PE 当作 nn.Parameter。
        self.register_buffer("pe",pe)
        return
        raise NotImplementedError("TODO: SinusoidalPositionalEncoding.__init__")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TODO：检查 x.shape == [B,L,d_model]，以及位置范围合法。
        if x.ndim != 3:
            raise ValueError()
        if x.shape[2] != self.pe.shape[1]:
            raise ValueError()
        if not isinstance(start_pos,int):
            raise TypeError()
        B,L,d_model = x.shape
        max_len = self.pe.shape[0]
        if start_pos < 0 or start_pos + L > max_len:
            raise ValueError("位置范围越界")
        
        # TODO：按 start_pos:start_pos+L 取位置编码，加到 x 上。
        pos = self.pe[start_pos:start_pos+L]
        pos_E = x + pos
        # TODO：若 valid_mask 不为 None，则要求形状 [B,L]，True 表示有效；
        #       把 Padding 位置的最终向量置为 0（教学工程约定）。
        if valid_mask is not None:
            if valid_mask.shape != pos_E.shape[:2]:
                raise ValueError()
            if valid_mask.dtype != torch.bool:
                raise ValueError()
            # unsqueeze 在指定维度插入一个大小为1的维度，-1表示末尾
            pos_E = pos_E.masked_fill(~valid_mask.unsqueeze(-1),0.0)
            # 把掩码为True的地方为0.0
        return pos_E
        raise NotImplementedError("TODO: SinusoidalPositionalEncoding.forward")


def make_valid_mask(token_ids: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """返回 [B,L] 的 bool 掩码：True=真实 Token，False=Padding。"""
    # TODO：检查 token_ids 为 [B,L]，返回 bool 掩码。
    if token_ids.ndim != 2:
        raise ValueError()

    # token_ids是token不是Embedding
    return token_ids != padding_idx
    raise NotImplementedError("TODO: make_valid_mask")


class TransformerInputEmbedding(nn.Module):
    """Token Embedding + Position Encoding；最终输出仍为 [B,L,d_model]。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 128,
        padding_idx: int = 0,
    ):
        super().__init__()
        # TODO：定义 self.token_embedding、self.position_encoding。
        self.token_embedding = TokenEmbedding(vocab_size,d_model,padding_idx)
        self.position_encoding = SinusoidalPositionalEncoding(d_model,max_len)
        # pe一开始使用的arange、zeros来注册，默认不计算梯度
        # TODO：保存 padding_idx，模型应只含 Token Embedding 可训练参数。
        self.padding_idx = padding_idx
        return 
        raise NotImplementedError("TODO: TransformerInputEmbedding.__init__")

    def forward(self, token_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # TODO：先查 Token Embedding，再加 PE。
        token = self.token_embedding(token_ids)
        mask = make_valid_mask(token_ids,self.padding_idx)
        PE = self.position_encoding(token,start_pos,mask)
        # TODO：调用 make_valid_mask，确保 Padding 位置最终输出为零。
        # PAD掩码，不是后面的因果掩码
        
        return PE
        raise NotImplementedError("TODO: TransformerInputEmbedding.forward")


def shift_right(target_ids: torch.Tensor, bos_id: int, padding_idx: int = 0) -> torch.Tensor:
    """教师强制训练：目标 [y1,y2,...] → Decoder 输入 [BOS,y1,...]。

    本函数只负责输入对齐；防止 Attention 看到未来位置，仍需 Day 17 的 Causal Mask。
    """
    # TODO：检查 target_ids 是 [B,T] 的整数 ID，T>0，bos_id 合法。
    if target_ids.ndim != 2 :
        raise ValueError()
    if target_ids.shape[1]<=0:
        raise ValueError()
    if target_ids.dtype not in (torch.int32,torch.int64):
        raise ValueError()
    if not isinstance(bos_id,int):
        raise TypeError()
    if bos_id <0:
        raise ValueError()
    
    # TODO：创建同形状的 LongTensor；首列放 BOS；其余列用 target_ids[:,:-1]。
    shifted = torch.zeros_like(target_ids,dtype=torch.long)
    shifted[:,0] = bos_id
    shifted[:,1:] = target_ids[:,:-1]
    return shifted
    # TODO：不要从目标序列中直接移除 PAD；保持长度和对齐关系。
    raise NotImplementedError("TODO: shift_right")
