"""Day 19：Transformer 子层与结构的学生 TODO 模板。

范围：原论文 Post-LN、Position-wise FFN、单层 Encoder/Decoder、词表输出投影。
依赖：用户在 Day 18 实现的 multihead_attention.py。
约定：batch_first；布尔 keep_mask=True 表示允许访问；模块中的 Dropout 随 train()/eval() 控制。
本文件不实现完整 Token Embedding、KV-Cache 或翻译训练循环。
"""
from __future__ import annotations
import torch
from torch import nn
from multihead_attention import MultiHeadAttention


class PostLayerNormResidual(nn.Module):
    """原论文的 Post-LN：LayerNorm(x + Dropout(sublayer_output))。"""

    def __init__(self, d_model: int, dropout_p: float = 0.0, eps: float = 1e-5) -> None:
        super().__init__()
        # TODO：验证维度和 Dropout 概率。
        if d_model <= 0 or dropout_p < 0 or dropout_p > 1.0:
            raise ValueError()
        # TODO：初始化 self.dropout、self.norm（nn.LayerNorm(d_model, eps=eps)）。
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(d_model,eps,elementwise_affine=True,bias=True)
        return
        raise NotImplementedError("TODO: PostLayerNormResidual.__init__")

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        # TODO：验证 x 和 sublayer_output 的 shape 一致。
        if x.shape != sublayer_output.shape:
            raise ValueError()
        # TODO：必须先残差相加，再做 LayerNorm；不要写成 Pre-LN。
        x = x+sublayer_output
        x = self.norm(x)
        return x
        raise NotImplementedError("TODO: PostLayerNormResidual.forward")


class PositionwiseFeedForward(nn.Module):
    """原论文结构：Linear(d_model,d_ff) -> ReLU -> Linear(d_ff,d_model)。

    此教学版本把残差路径上的 Dropout 放在 PostLayerNormResidual 中，
    为了便于逐位置数值测试，不在 FFN 内部再增加其他操作。
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # TODO：初始化 self.linear1、self.activation、self.linear2。
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.activation = nn.ReLU()
        return
        raise NotImplementedError("TODO: PositionwiseFeedForward.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO：沿最后一维依次执行 Linear -> ReLU -> Linear。
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        # TODO：FFN 不得在序列维度上进行混合。
        return x
        raise NotImplementedError("TODO: PositionwiseFeedForward.forward")


class TransformerEncoderLayer(nn.Module):
    """单层 Encoder：Self-Attention -> Add&Norm -> FFN -> Add&Norm。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        # TODO：创建 self.self_attn (Day18 的 MultiHeadAttention)。
        self.self_attn = MultiHeadAttention(d_model,num_heads,dropout_p)
        # TODO：创建 self.addnorm1、self.ffn、self.addnorm2。
        self.addnorm1 = PostLayerNormResidual(d_model,dropout_p)
        self.ffn = PositionwiseFeedForward(d_model,d_ff)
        self.addnorm2 = PostLayerNormResidual(d_model,dropout_p)
        return
        raise NotImplementedError("TODO: TransformerEncoderLayer.__init__")

    def forward(self, x: torch.Tensor, src_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x:[B,S,d_model]；src_keep_mask 可以是 [B,1,S] 或 [B,S,S]。"""
        # TODO：自注意力 Q/K/V 取自 x，保留给定的 src_keep_mask。
        output = self.self_attn(query=x,key=x,value=x,keep_mask=src_keep_mask)
        # TODO：对每个子层分别执行 Post-LN Add & Norm。
        output_1 = self.addnorm1(x,output)
        output_ffn = self.ffn(output_1)
        output_2 = self.addnorm2(output_1,output_ffn)
        return output_2
        raise NotImplementedError("TODO: TransformerEncoderLayer.forward")


class TransformerDecoderLayer(nn.Module):
    """单层 Decoder：Masked Self -> Cross -> FFN，各自带 Post-LN。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        # TODO：创建 self.self_attn、self.cross_attn、self.ffn。
        self.self_attn = MultiHeadAttention(d_model,num_heads,dropout_p)
        self.cross_attn = MultiHeadAttention(d_model,num_heads,dropout_p)
        self.ffn = PositionwiseFeedForward(d_model,d_ff)
        # TODO：分别创建 self.addnorm1、self.addnorm2、self.addnorm3。
        self.addnorm1 = PostLayerNormResidual(d_model,dropout_p)
        self.addnorm2 = PostLayerNormResidual(d_model,dropout_p)
        self.addnorm3 = PostLayerNormResidual(d_model,dropout_p)
        return
        raise NotImplementedError("TODO: TransformerDecoderLayer.__init__")

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_keep_mask: torch.Tensor | None = None,
        memory_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x:[B,T,d_model]，memory:[B,S,d_model]；输出:[B,T,d_model]。"""
        # TODO：第一子层使用 Decoder 的 x 计算 Causal Self-Attention。
        output_mask_attention = self.self_attn(query = x,key=x,value=x,keep_mask = tgt_keep_mask)
        output_mask_attention = self.addnorm1(x,output_mask_attention)
        # TODO：第二子层使用上一步 Decoder 隐藏状态作 Q，memory 作 K 和 V。
        output_cross_attention = self.cross_attn(query=output_mask_attention,key=memory,value=memory,keep_mask=memory_keep_mask)
        output_cross_attention = self.addnorm2(output_mask_attention,output_cross_attention)
        # TODO：第三子层使用逐位置 FFN。
        output_FFN = self.ffn(output_cross_attention)
        output_FFN = self.addnorm3(output_cross_attention,output_FFN)
        # TODO：注意每个子层都独立进行 Add & Norm。
        return output_FFN
        raise NotImplementedError("TODO: TransformerDecoderLayer.forward")


class TokenOutputProjection(nn.Module):
    """将 Decoder 隐藏状态投影为词表 logits；不在 forward 中调用 Softmax。"""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        if vocab_size<=0:
            raise ValueError()
        # TODO：创建 self.proj=nn.Linear(d_model,vocab_size)。
        self.proj = nn.Linear(d_model,vocab_size)
        return
        raise NotImplementedError("TODO: TokenOutputProjection.__init__")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # TODO：返回形状 [B,T,V] 的原始 logits。
        return self.proj(hidden_states)
        raise NotImplementedError("TODO: TokenOutputProjection.forward")

    def probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # TODO：仅用于展示/推理时按词表最后一维计算 Softmax。
        logits = self.forward(hidden_states)
        return torch.nn.functional.softmax(logits,dim=-1)
        raise NotImplementedError("TODO: TokenOutputProjection.probabilities")
