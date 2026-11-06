# Day 20｜KV-Cache 时间线与手算题

## 1. 时间线（B=2，H=2，d_k=4）

```text
输入 Token:          A       B       C       D       E
绝对 Position:       0       1       2       3       4
                   └──── Prefill ────┘  └─ Decode ─┘

Prefill:
  Q_new: [B,H,3,d_k]   K_new/V_new: [B,H,3,d_k]
  K_cache: [A,B,C]      V_cache: [A,B,C]
  Scores: [B,H,3,3]

Decode D:
  Q_new: [B,H,1,d_k]   K_new/V_new: [B,H,1,d_k]
  K_cache: [A,B,C,D]    V_cache: [A,B,C,D]
  Scores: [B,H,1,4]
  Query 全局位置 = 3，不是 0。

Decode E:
  Q_new: [B,H,1,d_k]   K_new/V_new: [B,H,1,d_k]
  K_cache: [A,B,C,D,E]  V_cache: [A,B,C,D,E]
  Scores: [B,H,1,5]
  Query 全局位置 = 4。
```

## 2. 动手画（必做）

请用手画出 Prefill 与第二次 Decode 的 Mask，写明行是 Query、列是 Key，True 表示允许读取。

## 3. 两个对照实验（必做）

1. 正确情况：固定权重、Dropout=0、所有位置使用正确绝对位置编码、Mask 按 Query 全局位置构造。
2. 错误情况：Decode 阶段错误地把新 Token 的位置编码设为 0。比较差异并解释原因。

> 注：只允许在正确情况中对数值等价要求 `1e-5`。错误情况是为了发现位置错位，不应强行要求其与全量前向一致。

## 4. 复杂度观察

给定长度 T 的序列，逐前缀重新运行投影的 Token 参与次数为

\[
1+2+\dots+T = T(T+1)/2
\]

使用缓存后，每个 Token 仅投影一次，合计 T 个位置。

**这只是 Q/K/V 投影参与的 Token 数对照，并非实际端到端运行时间的固定倍数。** 带 KV-Cache 时，新 Query 仍需与历史 Key 匹配，且 KV 内存随 T 增加。

对一个一般的标准多头 Transformer，假设 N 层、Batch=B、Head 数 H、每头维度 d_k、每个标量占 s 字节，标准 MHA 且 `d_v=d_k` 时：

\[
\text{KV Bytes}=2\,N\,B\,H\,T\,d_k\,s
\]

例如 N=2，B=2，H=2，T=5，d_k=4，float32 的 s=4，则 KV 缓存数据占 640 字节（不含 Python/Tensor 元数据及分配开销）。
