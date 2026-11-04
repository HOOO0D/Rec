# Day 17｜Scaled Dot-Product Attention 学习与手算记录

> 本文档需结合《Attention Is All You Need》§3.2.1 和 §3.2.3 自主填写；所有【待完成】都需要替换。

## 1. 核心公式与张量形状

- Q、K、V 各自代表什么：**【待完成】**
- Q `[B,Lq,dk]`、K `[B,Lk,dk]`、V `[B,Lk,dv]` 的含义：**【待完成】**
- 分数矩阵、权重矩阵、输出矩阵的形状：**【待完成】**
- 为什么 Key 和 Value 必须有相同的 Lk：**【待完成】**

## 2. 缩放为何使用 sqrt(dk)

- Q·K 随维度增加的数值变化与 Softmax 饱和问题：**【待完成】**
- 除以 sqrt(dk) 的直觉推导及其假设：**【待完成】**

## 3. 必须手算：1 个 Query，2 个 Key

已知 Q=[1,0]，K1=[1,0]，K2=[0,1]，V1=[2,0]，V2=[0,4]。

- QK^T：**【待完成】**
- 缩放分数：**【待完成】**
- Softmax 分子、分母及两个权重：**【待完成】**
- Value 加权输出：**【待完成】**
- 手算与程序输出的最大误差：**【待完成】**

## 4. Padding Mask 和 Causal Mask

- Key Padding 与 Query Padding 作用是否相同：**【待完成】**
- 4 个 token 的因果掩码（True=允许）：**【待完成】**
- 一行所有 Key 被屏蔽时采用的结果约定：**【待完成】**
- `scaled_dot_product_attention` 与 `MultiheadAttention.key_padding_mask` 的 True 语义：**【待完成】**
- Mask 应在 Softmax 前还是后执行？为什么：**【待完成】**

## 5. 三种 Attention 的 Q/K/V 来源

| 模块 | Q 的来源 | K、V 的来源 | 是否需要 Causal Mask |
|---|---|---|---|
| Encoder Self-Attention | 【待完成】 | 【待完成】 | 【待完成】 |
| Decoder Self-Attention | 【待完成】 | 【待完成】 | 【待完成】 |
| Encoder–Decoder Cross-Attention | 【待完成】 | 【待完成】 | 【待完成】 |

## 6. 实验记录

- PyTorch 版本 / CPU 或 CUDA：**【待完成】**
- A–D 自动测试得分：**【待完成】**
- 未来信息泄漏测试结果：**【待完成】**
- 遇到的问题、如何解决：**【待完成】**
- Day 18 要复用的函数：**【待完成】**
