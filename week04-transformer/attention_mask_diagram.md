# Day 17｜Attention 与 Mask 图解作业

> 请独立补齐。必须包含 True=允许访问 的图或矩阵。

## A. Encoder Self-Attention（可双向）

- Q 来源：**【待完成】**
- K/V 来源：**【待完成】**
- 若有三个有效 Token，允许访问矩阵：**【待完成】**
- Padding 的处理方法：**【待完成】**

## B. Decoder Self-Attention（Causal）

- Q 来源：**【待完成】**
- K/V 来源：**【待完成】**
- 4x4 下三角掩码（True=允许，False=禁止）：**【待完成】**
- 为什么 Shifted Right 不足以独立防泄漏：**【待完成】**
- 改变未来 Token 后，前三个输出是否改变：**【待完成】**

## C. Cross-Attention（Encoder→Decoder）

- Q 来源：**【待完成】**
- K/V 来源：**【待完成】**
- 若 Decoder 长度 2、Encoder 长度 4，权重形状：**【待完成】**
- 为什么 Cross-Attention 通常屏蔽 Encoder 的 Padding，但不对 Encoder 序列使用 Decoder 式 Causal Mask：**【待完成】**

## D. 代码约定对照

- 本项目的布尔 Mask：True=允许；False=禁止。
- PyTorch `scaled_dot_product_attention` 的 True 含义：**【待完成】**
- PyTorch `MultiheadAttention` 的 `key_padding_mask` True 含义：**【待完成】**
- 全被屏蔽的 Query 行约定：**【待完成】**
