# Day 18｜Multi-Head Attention 阅读与实验记录

> 本文件需独立填写。替换所有 TODO 后执行 `Test_Day18.py`。

## A. 原论文 Section 3.2.2

- TODO：用 150 字概括多头注意力相对于单头的优势、局限。
- TODO：抄写并解释 MultiHead 与 head_i 的公式（用自己的话）。
- TODO：原论文 h=8、d_model=512、d_k=d_v=64 的关系。

## B. 四组投影矩阵

- W_Q：TODO：作用、输入和输出 Shape。
- W_K：TODO：作用、输入和输出 Shape。
- W_V：TODO：作用、输入和输出 Shape。
- W_O：TODO：作用、输入和输出 Shape。
- TODO：d_model=8,h=2,bias=True 时模型为什么有 288 个可训练参数。
- TODO：说明若设置 bias=False，参数量会如何变化。

## C. 张量形状全过程

- 输入：TODO
- 投影：TODO
- 拆头：TODO
- 注意力权重：TODO
- 每头输出：TODO
- 拼头：TODO
- W_O 后：TODO

## D. Self-Attention 与 Cross-Attention

- Self-Attention Q/K/V 来源：TODO
- Cross-Attention Q/K/V 来源：TODO
- 为什么两者的 Query/Key 长度关系不同：TODO
- Decoder Self-Attention 为什么需要 Causal Mask：TODO
- Day 17 布尔掩码与 `nn.MultiheadAttention` 布尔掩码语义差别：TODO

## E. 数值实验与复盘

- 测试得分：TODO
- 演示 Self/Cross 形状：TODO
- 未来 Token 泄漏误差：TODO
- 梯度测试说明：TODO
- 今日最容易犯的三个错误：TODO
- Day 19 需要复用什么接口：TODO
