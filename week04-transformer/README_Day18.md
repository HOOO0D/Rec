# Day 18｜Multi-Head Attention 练习说明

## 0. 先决条件

已经通过 Day 17 并拥有 `scaled_attention.py`，尤其是 `scaled_dot_product_attention()` 和 `make_causal_keep_mask()`。
本压缩包不包含 Day 17 的学生解答，也不会覆盖已有文件。

## 1. 文件复制目录

```text
E:\AI\rec-learning\week04-transformer\
    embeddings_position.py          # Day 16（保留）
    scaled_attention.py             # Day 17（必须已完成）
    multihead_attention.py          # Day 18 TODO
    Test_Day18.py                   # Day 18 测试
    run_day18.py                    # Day 18 演示
    Day18_notes.md                  # 理论记录
    attention_heads_diagram.md      # 结构图
```

## 2. 实现顺序

1. `split_heads`、`combine_heads`：严格检查形状和元素顺序。
2. `expand_keep_mask`：保留 Day 17 的 True=允许访问约定，处理 Head 维广播。
3. `MultiHeadAttention.__init__`：定义 4 个独立 `nn.Linear` 投影模块。
4. `MultiHeadAttention.forward`：实现 Self/Cross 模式；复用 Day 17 的 SDPA。
5. 填写理论笔记，再跑自动测试。

教学要求：模型使用 batch_first 输入；若选择 bias=True，输出投影会对全屏蔽 Query 行引入 bias，故需要在投影后再次显式置零这些行。

## 3. 运行

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week04-transformer

python .\Test_Day18.py --skip-docs
python .\Test_Day18.py --verbose
python .\run_day18.py --device auto --output-dir .\experiments\day18_demo
```

- A–E：独立代码正确性，满分 90。
- F：理论笔记和结构图完整性，10 分。自动脚本不评价文字内容的学术准确性，需要自行核对。
- 演示脚本生成 `multihead_demo.pt` 与 `multihead_summary.json`，为未训练模型的前向数值检查结果。

## 4. 参数约定与边界

- `d_model>0`，`num_heads>0` 且 `d_model%num_heads==0`。
- 拆头输出 `[B,H,L,d_model/H]`，拼头输出 `[B,L,d_model]`。
- Self 模式只传 query；Cross 模式同时传 query、key、value。
- Query/Key/Value 的 Batch、特征维及 Key/Value 长度要一致，Query 长度可以不同。
- `keep_mask`: True=允许访问，False=屏蔽。
- 所有 Head 都被屏蔽的 Query 行最终输出必须为零。
- 训练模式可启用 attention dropout；`eval()` 不启用。

## 5. Day 19 的衔接

Day 19 将把 MHA 接入 Encoder/Decoder 的 Add & Norm 与 FFN，原论文使用 Post-LN。
