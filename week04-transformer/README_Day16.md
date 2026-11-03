# Day 16｜Transformer 输入表示练习包

依据 `RoadMap(2).md` 的 Day 16：论文架构总览 + Token Embedding + Position + Shifted Right。

## 今日范围

- 精读原论文 Abstract、Section 1–2、Section 3.1、3.4、3.5 及 Figure 1。
- 实现 Token Embedding（含 `sqrt(d_model)`）、固定正弦位置编码、Padding、Decoder Shifted-Right 输入。
- Q/K/V 今天仅做概念预习，完整注意力和 Causal Mask 在 Day 17。

## 文件

- `embeddings_position.py`：中文注释 TODO，须自行补全。
- `Test_Day16.py`：A–F 共 100 分自动验收。
- `run_day16.py`：完成 TODO 后进行一次 Encoder/Decoder 输入演示。
- `transformer_architecture.md`：填写 Figure 1 架构数据流。
- `Day16_notes.md`：论文阅读、手算、SASRec 迁移笔记。

## Windows PowerShell 运行

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week04-transformer

# 完成 embeddings_position.py TODO 后，先运行代码测试（90 分）
python .\Test_Day16.py --skip-docs

# 同时完成两份 Markdown 作业后，运行完整 100 分测试
python .\Test_Day16.py

# 演示 Encoder 与 Decoder 的输入表示，并保存产物
python .\run_day16.py --device auto --output-dir .\experiments\day16_demo
```

## 约定和注意事项

1. 词表 Token ID 从 0 开始，0=PAD，1=BOS，2=EOS。
2. 输入 ID 形状 `[B,L]`，向量形状 `[B,L,d_model]`。
3. `nn.Embedding(..., padding_idx=0)` 令 PAD Embedding 行不从查表操作接收梯度。
4. 原论文将 Token Embedding 与 PE 相加；本练习额外约定最终 Padding 位置整行为 0，方便 Day 17 集成，但该约定不代表注意力 Padding Mask 已经实现。
5. 原论文采用固定正弦 PE；本练习 `SinusoidalPositionalEncoding.pe` 使用 Buffer，不能将其视作训练参数。
6. `shift_right()` 仅创建教师强制的 Decoder 输入，仍需 Day 17 Causal Mask 才能防止 Attention 读取未来 token。
7. 练习版要求 `d_model` 是正偶数，简化 sin/cos 配对；这一约束不意味着所有 Transformer 都必须具有偶数维度。
8. 原论文共享 Source Embedding、Target Embedding 和预测输出层参数；Day 16 只练习输入层，因此演示脚本分别创建 Source / Target Embedding，不测试参数共享。
9. 今日不下载或训练机器翻译模型。输入序列均为人工构造。

## 评分

A Token Embedding 15；B 正弦位置编码 25；C 输入融合和 Padding 20；D Shifted Right 20；E 数值梯度及异常边界 10；F 架构图和笔记 10。合计 100 分。

Day 16 完成条件：自动测试 100/100、能独立解释 Figure 1 输入数据流。Day 17 再实现 Scaled Dot-Product Attention 和 Mask。

## 阅读材料

- https://arxiv.org/html/1706.03762
- https://nlp.seas.harvard.edu/annotated-transformer/
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
