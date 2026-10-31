# Day 13｜LightGCN 多层传播与模型前向（独立实现包）

## 一、任务要求

按照 RoadMap(2).md 的 Day 13，独立实现 `E[k+1]=A_norm @ E[k]`、层融合、评分与梯度回传。
**本日不训练 BPR，不计算 Recall/NDCG，不做不同层数性能消融。**

## 二、文件组成

- `lightgcn_model.py`：待完成的 `LightGCN(nn.Module)` 模板，**所有注释均为中文**。
- `Test_Day13.py`：自动验收，A–E 80 分、F 20 分。
- `run_day13.py`：读取 Day 11 保存的归一化训练图，完成前向并保存实验产物。
- `Day13_notes.md`：公式、手算题、Shape、理论思考题与记录模板。

## 三、建议项目结构

```text
E:\AI\rec-learning\week03-lightgcn\
├── graph_basics.py              # Day 10
├── graph_aggregation.py         # Day 11
├── lightgcn_model.py           # Day 13
├── Test_Day13.py
├── run_day13.py
├── Day13_notes.md
└── experiments\
    ├── day10_ml100k\
    │   ├── adjacency.pt
    │   ├── degrees.pt
    │   └── graph_summary.json
    ├── day11_ml100k\
    │   ├── normalized_adjacency.pt
    │   ├── inverse_sqrt_degrees.pt
    │   └── normalization_summary.json
    └── day13_ml100k\
```

## 四、运行

在 Windows 11 的 PowerShell 中：

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week03-lightgcn

# 合成图单测：A–E 共 80 分
python .\Test_Day13.py

# 使用真实 Day 11 图执行完整 100 分验收
python .\Test_Day13.py --graph-dir .\experiments\day11_ml100k

# 使用第 11 天归一化图进行正式前向，保留审计文件
python .\run_day13.py `
    --graph-dir .\experiments\day11_ml100k `
    --output-dir .\experiments\day13_ml100k `
    --embedding-dim 16 `
    --n-layers 2
```

**注意：** 先完成 `lightgcn_model.py` 全部 TODO，再执行自动测试。未传入 `--graph-dir` 时，最高只能取得 80/100。

## 五、输出

- `experiments/day13_ml100k/forward_summary.json`：输入图来源、用户/物品数、参数量、模型维度、是否训练等记录。
- `experiments/day13_ml100k/forward_sample.pt`：前五位用户的确定性前向评分和 Embedding 样例（**未训练的随机初始化模型**）。

请勿将这些随机初始化分数当作实验推荐指标。Day 14 才连接 BPR 训练；Day 15 才做不同 K 的固定协议对比。

## 六、验收说明

A 初始化与参数约束（10 分）；B 0 层 MF 退化（15 分）；C 2 层手算传播（25 分）；D 层权重与评分接口（15 分）；E 梯度与孤立节点（15 分）；F 真实训练图集成（20 分）。

通过建议：A–F 全通过且手算笔记与代码接口一致。
