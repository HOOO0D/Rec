# Day 20｜KV-Cache 与自回归增量解码

## 今日范围
- 学习 Prefill、Decode、逐层 K/V、绝对位置偏移和数值一致性。
- 不做 PagedAttention、FlashAttention、完整翻译系统、复杂 Padding Batch 或 Transformer 解码策略。
- **只在推理模式使用缓存**，输入已经加入 Token/Position Embedding。
- Day 16 的位置编码用于演示；Day 17/18 的 Attention 是本日核心代码依赖。

## 文件
- `kv_cache.py`：学生 TODO（需独立完成）
- `Test_Day20.py`：A–E 90 分的代码测试，F 10 分的理论笔记检查
- `run_day20.py`：固定模型执行 Prefill/Decode，与完整前向比较；输出日志文件
- `Day20_notes.md`：理论、形状手算和实验记录
- `cache_timeline.md`：Prefill/Decode 时间线与复杂度练习
- `README_Day20.md`：本文件

## 推荐目录

将文件放入 `E:\AI\rec-learning\week04-transformer\`，与已完成的以下文件放在同级：
- `embeddings_position.py`（Day 16）
- `scaled_attention.py`（Day 17）
- `multihead_attention.py`（Day 18）

命令：

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week04-transformer
python .\Test_Day20.py --skip-docs
python .\run_day20.py --output-dir .\experiments\day20_demo
# 填写 Day20_notes.md 后
python .\Test_Day20.py
```

自动评分：
- A 缓存结构与边界：20
- B 增量因果掩码与绝对位置：15
- C 单层 Prefill/Decode：25
- D 两层独立缓存与数值一致性：20
- E 位置编码对齐、推理约束、统计：10
- F 理论笔记：10

全套 100 分为目标；A–E 代码不能有失败项，最大数值误差 <=1e-5。
理论笔记的自动校验只检查完成性，正确性须结合答案和老师反馈审核。

## 额外提醒

1. `is_causal=True` 的非方形矩阵对齐规则不能直接当作带历史缓存的增量 Mask；按照 Query 的全局位置计算。
2. Prefill 可以是多个 Token，Decode 只输入新 Token；缓存是 K/V，不是注意力权重，也不是隐藏状态的通用缓存。
3. 只有输入位置编码、所有模型权重、Mask、Dropout 状态完全一致时，比较缓存与非缓存输出才有意义。
4. 演示使用无 Padding 的固定长度玩具序列，避免将本课扩展为复杂动态 Batch 工程。
5. 对于缓存了完整 Prefill 的推理，在下一次调用中**不得再次传入整个前缀**，否则会重复追加 K/V。

## 资料
- https://huggingface.co/docs/transformers/en/cache_explanation
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://arxiv.org/abs/1706.03762
