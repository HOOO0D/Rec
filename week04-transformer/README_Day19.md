# Day 19｜Add & Norm、FFN、Encoder/Decoder 子层练习包

## 0. 必要前提

- 已完成 Day 17 的 `scaled_attention.py` 和 Day 18 的 `multihead_attention.py`。
- 采用 Day 18 的 `MultiHeadAttention` 类接口：`MultiHeadAttention(d_model,num_heads,dropout_p)`；`forward(query,key=None,value=None,keep_mask=None)`；`True` 代表允许访问。
- Python 3.10+、PyTorch；无需 RecBole / Transformers / PyTorch Geometric。

## 1. 文件

- `transformer_layers.py`：**学生版 TODO**，包含 PostLayerNormResidual、PositionwiseFeedForward、TransformerEncoderLayer、TransformerDecoderLayer、TokenOutputProjection。
- `Test_Day19.py`：A–F 六组测试，共 100 分。
- `run_day19.py`：随机输入下的单层 Encoder/Decoder 与词表输出投影演示，不进行训练。
- `Day19_notes.md`：理论笔记和手算模板。
- `day19_structure.md`：本日专用结构图作业，不覆盖 Day 16 的 `transformer_architecture.md`。
- `README_Day19.md`：本文件。

## 2. 放置文件

把六个文件复制到 `E:\AI\rec-learning\week04-transformer`，不要覆盖已经实现的 Day 16–18 文件。

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week04-transformer

# 完成 transformer_layers.py 的 TODO 后检查代码（最高 90 分）
python .\Test_Day19.py --skip-docs

# 填写 Day19_notes.md 和 day19_structure.md 后完整验收
python .\Test_Day19.py

# 出错时查看完整 Traceback
python .\Test_Day19.py --verbose

# 演示前向过程
python .\run_day19.py --device auto --output-dir .\experiments\day19_demo
```

输出：`experiments/day19_demo/day19_forward.pt` 与 `day19_summary.json`。

## 3. 100 分验收

| 模块 | 检查 | 分值 |
|---|---|---:|
| A | Post-LN 残差、LayerNorm、参数与 Pre-LN 差异 | 15 |
| B | FFN 数值/维度、逐位置不混合、梯度 | 20 |
| C | Encoder 两个子层、Mask 传递、反向传播 | 20 |
| D | Decoder 三个子层、因果约束、Cross-Attention、梯度 | 20 |
| E | 输出投影、Softmax、CrossEntropyLoss、端到端前向 | 15 |
| F | 本日笔记及结构图完整性 | 10 |
| **合计** | | **100** |

建议今日通过线 90 分，且 A–E 不允许跳过。F 的自动校验仅检测文件填写完整性，数学解释仍需自查。

## 4. 防止常见错误

- 原论文 **Post-LN**：`LayerNorm(x + Dropout(Sublayer(x)))`；不是 Pre-LN。
- `nn.LayerNorm(d_model)` 沿**每个 Token 的最后一维**归一化，不跨 Batch 或时间维度。
- FFN 仅沿特征维操作；Attention 才聚合不同 Token 的信息。
- Decoder 的第 1 子层用 Causal Mask；第 2 子层的 Q 来自 Decoder，K/V 来自 Encoder。
- 词表输出 `forward()` 返回原始 logits；`F.cross_entropy()` 不要求你提前 Softmax。
- 今天只完成**一层**模块的可验证前向，不要求训练翻译模型；完整多层整合为 Day 21 的任务。
