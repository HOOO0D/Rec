# Day 18｜Multi-Head Attention 结构图作业

要求绘制并解释两幅结构图；图可以采用 ASCII/Markdown/Mermaid，图下方写维度注释。

## 图 1：Self-Attention

TODO：在这里绘制从 X 到 W_Q/W_K/W_V、Split Heads、SDPA、Concat、W_O 的完整数据流，并且标注输入和输出的形状；至少标明 B=2、L=4、d_model=8、h=2。

## 图 2：Cross-Attention

TODO：在这里绘制从 Decoder Query 和 Encoder Key/Value 到投影、拆头、权重、输出的完整数据流；要求明确 L_q=3、L_k=5 时的权重形状。

## Mask 分析

TODO：画出一个长度 4 的 Decoder Causal Mask，说明广播到 [B,H,4,4] 的规则。

## 解释

TODO：为什么不能把 [B,L,H,d_k] 直接 view 成 [B,L,H*d_k] 来做正确的 Concat？请说明 transpose 的必要性。
