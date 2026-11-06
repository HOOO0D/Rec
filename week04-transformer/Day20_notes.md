# Day 20：KV-Cache 理论与实验记录

## 1. 阅读资料
- [ ] Hugging Face Caching：Attention Matrices / Cache Storage / Cache Position
- [ ] Transformer 论文 Section 3.2.1 和 3.2.3
- [ ] PyTorch 非方形 `is_causal` 掩码行为

## 2. 解释以下概念（每题写 2–5 句话）

### 2.1 Prefill 与 Decode 的区别
TODO

### 2.2 为什么保存历史 K/V、不需要保存历史 Q？
TODO

### 2.3 为什么 KV-Cache 不应该直接当作训练参数？训练中为什么通常禁用缓存？
TODO

### 2.4 每层为什么要维护独立 K/V？
TODO

### 2.5 什么是绝对位置偏移？Prefill 3 个 Token 后新 Token 的位置是多少？
TODO

### 2.6 `Q_len=1,K_len=4,query_start=3` 应该得到什么 Mask？为什么不能盲用非方形 `is_causal=True`？
TODO

### 2.7 KV-Cache 减少了哪些重复计算？又会增加哪些显存开销？
TODO

## 3. 手算与形状（必做）

假设 B=2，H=2，d_k=4，Prefill 长度为 3，后续生成两个 Token。

| 阶段 | 新 Query shape | 新 K/V shape | Cache K/V shape | Attention 权重 shape |
|---|---|---|---|---|
| Prefill | TODO | TODO | TODO | TODO |
| Decode 1 | TODO | TODO | TODO | TODO |
| Decode 2 | TODO | TODO | TODO | TODO |

## 4. 数值一致性实验（必做）
- torch / device / dtype：TODO
- 固定种子：TODO
- 层数、Head 数、d_model：TODO
- Prefill 长度：TODO
- 不缓存的前向输出 shape：TODO
- 使用缓存的增量输出 shape：TODO
- 最大绝对误差：TODO
- 是否 <= 1e-5：TODO

## 5. 效率与空间分析（必做）
- 不缓存对每个逐步增长的完整前缀重复计算 K/V 投影时，参与投影的 Token 数之和：TODO
- 使用缓存时，每个 Token 仅投影一次，对应总数：TODO
- 模型层数为 N、Batch B、Head H、当前缓存长度 T、head_dim d_k、元素字节数 s：KV 缓存理论大小：TODO
- 一组实际测量 / 理论计算及说明：TODO

## 6. 与 Day 21 / SASRec 衔接
- Day 21 在完整 Decoder Block 中应该如何使用每层缓存：TODO
- 为什么标准离线 SASRec 可以先不实现 KV-Cache：TODO
