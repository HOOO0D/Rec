# Day 16｜Attention Is All You Need 输入层学习笔记

## A. 论文背景（Section 1–2）

- RNN 的序列计算限制：【待填写】
- Transformer 不依赖循环结构，为什么仍能处理顺序：【待填写】

## B. Token Embedding（Section 3.4）

- 词表 ID 如何变成向量：【待填写】
- 为什么乘 sqrt(d_model)：【待填写】
- `nn.Embedding` 的 padding_idx 行为什么没有常规梯度：【待填写】

## C. Positional Encoding（Section 3.5）

- 正弦/余弦公式：【待填写】
- 手算 PE(0,0)、PE(0,1)、PE(1,0)、PE(1,1)：【待填写】
- 为什么必须有位置编码：【待填写】
- 原论文固定位置编码与 SASRec 常见可学习位置 Embedding 的异同：【待填写】

## D. Decoder Shifted Right

- 目标序列：[5,7,9,2,0]
- BOS=1 时 Decoder 输入：【待填写】
- Shifted Right 为什么仍不能代替 Day 17 的 Causal Mask：【待填写】

## E. Q/K/V 预习（只需概念）

- Q（Query）是什么：【待填写】
- K（Key）是什么：【待填写】
- V（Value）是什么：【待填写】
- Encoder Self-Attention、Decoder Masked Self-Attention、Cross-Attention 的 Q/K/V 来源：【待填写】

## F. 代码与测试记录

- torch 版本：【待填写】
- 运行设备：【待填写】
- 自动测试得分：【待填写】
- 失败项和原因：【待填写】
- 今日尚未解决的问题：【待填写】
