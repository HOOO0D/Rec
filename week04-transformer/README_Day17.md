# Day 17｜Scaled Dot-Product Attention + Padding/Causal Mask

依据 `RoadMap(2).md` 的 Day 17：精读《Attention Is All You Need》§3.2.1、§3.2.3 和 Figure 2 左图，独立实现缩放点积、Softmax、Value 聚合、Padding/Causal Mask。

## 文件

- `scaled_attention.py`：中文注释的 TODO 模板（请独立填写）。
- `Test_Day17.py`：A–E 100 分自动验收；`--skip-docs` 检查前 90 分代码。
- `run_day17.py`：人工 Q/K/V 的单头 Attention 演示，无模型训练。
- `Day17_notes.md`：论文公式、手算与实验记录。
- `attention_mask_diagram.md`：Encoder/Decoder/Cross-Attention 与 Mask 图解任务。

## 复用 Day 16

Day 16 的 `TransformerInputEmbedding` 已能从 token ID 得到 `[B,L,d_model]`。今天可以令 `Q=K=V=X` 理解单头 Self-Attention；**但完整 Transformer 中，Q/K/V 通常经过独立的可训练线性投影，安排在 Day 18**。

## 约定

1. `keep_mask=True` 表示允许注意，`False` 表示屏蔽；与 PyTorch `F.scaled_dot_product_attention` 的 bool attn_mask 一致，但与 `nn.MultiheadAttention.key_padding_mask` 的语义相反。
2. `make_padding_keep_mask(token_ids, for_query=False)` 返回 `[B,1,L]`，屏蔽 Padding Key；`for_query=True` 返回 `[B,L,1]`，令无效 Query 行输出零。
3. `make_causal_keep_mask(L)` 返回 `[L,L]` 下三角布尔矩阵。
4. 先将屏蔽位置在 Score 上设为 `-inf`，再执行 Softmax；对整行均被屏蔽的情形，必须先安全处理，不允许 softmax 全 `-inf` 产生 NaN。最终规定该行权重和输出为零。
5. 本日不实现 `W_Q/W_K/W_V/W_O`，也不训练 Transformer。
6. `return_weights=True` 时需同时返回 `[B,Lq,dv]` 输出和 `[B,Lq,Lk]` 权重。
7. Day 20 才正式研究 KV-Cache；本日 `query_start` 仅作维度和位置约定预留。

## Windows PowerShell 运行

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week04-transformer

# 完成 TODO 后先检查代码（90 分）
python .\Test_Day17.py --skip-docs

# 填写 Markdown 作业后完整验收（100 分）
python .\Test_Day17.py

# 演示并生成保存文件
python .\run_day17.py --device auto --output-dir .\experiments\day17_demo
```

运行结果存储在 `experiments/day17_demo/attention_demo.pt` 与 `attention_summary.json`。

## 评分结构

| 项目 | 分数 |
|---|---:|
| A. 手算及 Q/K/V 维度 | 20 |
| B. Padding/Causal Mask、广播 | 25 |
| C. 未来信息不泄漏 | 20 |
| D. 全屏蔽行、梯度、非法输入 | 25 |
| E. 两份 Markdown 理论作业 | 10 |
| 总计 | 100 |

> 说明：E 仅自动检查文件是否完成填写，不自动判定所有文字的理论正确性；仍须独立口头解释并人工复核。建议 A–E 全部通过后再进入 Day 18。

## 官方资料

- 原论文：https://arxiv.org/abs/1706.03762
- Harvard Annotated Transformer：https://nlp.seas.harvard.edu/annotated-transformer/
- PyTorch SDPA：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
