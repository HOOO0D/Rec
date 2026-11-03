# Day 16｜Transformer Figure 1 架构与数据流标注

> 要求：参考原论文 Figure 1，独立补齐下列结构。今天不要求实现内部 Attention。

## 1. Encoder 左侧路径

输入->输入Embedding+位置编码->多头注意力->add&norm->FFN->add&norm->经N层处理后输出

## 2. Decoder 右侧路径

输入->输入Embedding+位置编码->带mask的多头注意力->add&norm->交叉注意力->add&norm->FFN->add&norm->经过N层->线性表示->softmax->输出

## 3. Encoder 和 Decoder 的输入有何不同？

Decoder的输入要右移一个token

## 4. Decoder 为什么使用 Shifted Right？

保证自回归特性，防止decoder提前看到Encoder的token

## 5. 为后续 Day 17 标注三种 Attention（只说明来源、功能，不实现）

【待填写】
- encoder-self-attention
- decoder-self-attention
- encoder-decoder-cross-Attention

## 6. 输出端的 Linear + Softmax 如何与词表对应？

linear计算表中的logits，Softmax转换成归一化概率
