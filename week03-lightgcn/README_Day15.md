# Day 15｜BPR-MF 与 LightGCN 公平对照、层数消融、阶段验收

## 本日定位

与 `RoadMap(2).md` 的 Day 15 对齐。Day 13 已实现 LightGCN 模型前向，Day 14 已实现 BPR 训练与验证。本日**不重写训练器**，只做 K=0–4 的统一协议比较、验证集选择以及一次最终测试。

**关键决策：** 把同一个 LightGCN 类的 `n_layers=0` 视为 BPR-MF 基线，这样模型结构退化为 Embedding 内积且可复用相同训练、初始化、优化器与评价协议。Day 7 原 SGD 结果仅供参考，不自动视为公平对照。

## 文件

| 文件 | 用途 | 需要独立完成 |
|---|---|---|
| `day15_analysis.py` | A–E 五个公平审计、选模与消融分析 TODO | **是** |
| `Test_Day15.py` | 自动测试：合成单测 80 分 + 真实实验审计 20 分 | 否 |
| `run_day15.py` | 先训练/验证（sweep），再最终测试（final）的运行器 | 否 |
| `Day15_notes.md` | 实验协议、消融分析、Week 3 复盘模板 | **是：填写** |
| `README_Day15.md` | 本说明 | 否 |

## 前置文件

将以上文件复制到你的 `E:\AI\rec-learning\week03-lightgcn\` 目录，与下列**已完成**代码放在同一目录：

```text
lightgcn_model.py        # Day 13
lightgcn_train.py        # Day 14
run_day14.py             # Day 14
run_day15.py             # Day 15
Test_Day15.py            # Day 15
```

必须拥有 Day 7 冻结目录：`config.json`、`train_pairs.tsv`、`valid_pairs.tsv`、`test_pairs.tsv`，以及 Day 11 目录：`normalized_adjacency.pt`、`normalization_summary.json`。所有训练图只能含训练交互。

## 执行步骤（PowerShell）

```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week03-lightgcn

# 1. 完成 day15_analysis.py 的五个 TODO，再做独立合成单测
python .\Test_Day15.py

# 2. 训练 K=0,1,2,3,4，按验证 NDCG@10 选择最优层数
#    day7_sgd_ml100k 仅是目录名示例，按你的实际路径修改。
python .\run_day15.py `
    --stage sweep `
    --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k `
    --graph-dir .\experiments\day11_ml100k `
    --output-dir .\experiments\day15_ml100k `
    --embedding-dim 16 --batch-size 1024 `
    --epochs 10 --learning-rate 0.001 `
    --reg-lambda 0.0001 --topk 10 `
    --seed 20261031 --device cuda

# 3. 在看完并记录五组验证结果之后，执行唯一一次最终测试
python .\run_day15.py `
    --stage final `
    --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k `
    --graph-dir .\experiments\day11_ml100k `
    --output-dir .\experiments\day15_ml100k `
    --device cuda

# 4. 真正的 Day 15 完整验收（需已有最终测试文件）
python .\Test_Day15.py `
    --run-dir E:\AI\rec-learning\week02-bpr\runs\day7_sgd_ml100k `
    --graph-dir .\experiments\day11_ml100k `
    --result-dir .\experiments\day15_ml100k
```

如果 PyTorch 无可用 CUDA，`sweep` 与 `final` 两条命令中的 `--device` 均改为 `cpu`。为了减少 GPU 与 CPU 比较引入的数值差异，五组训练应在同一设备执行。

**模型选择规则：** 每个 K 先由验证 NDCG@10 选 Epoch，再由五组最佳验证 NDCG@10 选择 K。若并列，选较浅的层数。训练日志输出 `K0` 至 `K4` 五个目录、`validation_comparison.csv` 和 `selection.json`。选择阶段不会输出测试指标。

**最终测试规则：** 仅评价 K=0 和选出的最优 K（若最优 K=0 则只评价一次），测试候选排除 Train∪Valid 的历史物品，**但训练图始终只包含 Train 边**；测试标签仅来自 Test。输出 `final_test_results.json`，再次执行 final 会拒绝，以免反复查看测试结果进行调参。

**注意：** 如果最终测试前发现 Bug，停止使用测试集，修正训练/评价协议并重新在新的实验目录运行整个 sweep。不要根据测试结果选择层数、学习率或早停轮数。

## 实验目录

```text
experiments/day15_ml100k/
├── K0/                                 # 统一实现的 MF 基线
│   ├── experiment_config.json
│   ├── epoch_metrics.csv
│   ├── best_valid_model.pt
│   └── run_summary.json
├── K1/
├── K2/
├── K3/
├── K4/
├── validation_comparison.csv           # 5 组验证指标及相对 K0 增益
├── selection.json                     # 最优层数及共享协议
└── final_test_results.json            # 最终阶段才生成
```

## 避免误读结果

- **K=0 ≡ MF** 是本次模型数学形式下的同实现对照；Day 7 SGD 不是同配置的控制组，不能直接比较。
- 五组都采用相同初始化种子和采样种子，有助于减少随机差异，但单个种子不能证明提升稳定；条件允许时补 3 个不同种子报告均值和标准差。
- 层数提升不一定使指标单调提高；不能看到 K=4 较差就直接断言“过平滑已被证明”，还要考虑优化、数据稀疏性和收敛程度。
- Day 14 的 10 Epoch 仅供教学起步；不能简单将这些数值视为论文性能。
- RecBole 等框架可能采用不同的 ID、采样、正则、划分和评价协议，应做实验协议审计。
- 不能与后续 SASRec 的时序下一物品推荐指标直接比较。
