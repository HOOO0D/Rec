# Day 15 独立对比与多学习率搜索

`day15_experiments.py` 复用现有 `run_day15.py`、训练器和 CSV 读取函数，不修改原实现。
以下命令在 `E:\AI\rec-learning` 执行，使用已安装 PyTorch、matplotlib 的环境。

## 1. 直接绘制已有 Day 15 结果

```powershell
python .\week03-lightgcn\day15_experiments.py --stage plot `
    --result-dir .\week03-lightgcn\experiments\day15_ml100k
```

图表默认写入结果目录的 `reports`，可用 `--report-dir` 指定其他目录：

| 文件 | 内容 |
|---|---|
| `k0_k4_curves.png` / `.svg` | K0–K4 的 BPR Loss、总 Loss、原始 L2、训练 pair accuracy、验证 Recall/Hit/NDCG |
| `mf_vs_best_curves.png` / `.svg` | 同一 lr 下 MF 与最佳 K 的上述逐轮曲线 |
| `mf_vs_best_at_selected_epochs.png` | 两个模型各自最佳验证轮次的训练 Loss 与验证指标 |
| `mf_vs_best_test.png` | 仅在已有 final 结果时生成，展示保存的测试 Recall/Hit/NDCG |
| `best_epochs.csv` | 五组各自最佳轮次的完整数值，以及相对同 lr K0 的指标增益 |
| `mf_vs_best.csv` | MF 与最优 K 的训练/验证对照；有 final 时附测试指标 |

曲线上的点标记该模型验证 NDCG 最优的 Epoch。Loss 来自该轮训练过程中按样本加权的平均值，
不是对保存权重重新计算的 Loss。现有运行器没有验证或测试 Loss，图中不会伪造这些数值。
原始 L2 尚未乘 `reg_lambda`。最佳 K 允许是 0，此时只展示 MF 一组，不重复绘制。

## 2. 多 lr × K 搜索

```powershell
python .\week03-lightgcn\day15_experiments.py --stage sweep `
    --lrs 0.0003 0.001 0.003 `
    --epochs 50 --device auto `
    --output-dir .\week03-lightgcn\experiments\day15_lr_sweep
```

默认使用现有 `week02-bpr/runs/day7_sgd_ml100k` 和 `week03-lightgcn/experiments/day11_ml100k`。
可用 `--run-dir`、`--graph-dir` 替换。`auto` 有 CUDA 就使用 CUDA，否则用 CPU。
默认会产生 3 × 5 = 15 次训练，每次 50 轮；不会自动启动最终测试。

每个 lr 使用独立子目录，例如 `lr_0.001/K0` 至 `K4`。再次执行完全相同的命令，
原运行器会核对配置并复用完整的 K 训练记录；不完整的单个 K 不支持断点续训，应换输出目录。
搜索网格及配置保存在 `sweep_config.json`，更改 lr 网格、轮数预算等配置需要新目录。

搜索顺序：每个 (lr,K) 根据验证 NDCG 选择最佳 Epoch，再对全部候选选择最佳 (lr,K)。
NDCG 并列时先选较小 K，再选较小 lr；Epoch 并列时保留首次达到最优值的轮次。
所有候选共享数据划分、图、随机种子、训练轮数预算和其余超参数。
MF 对照取选中 lr 下的 K0，便于控制变量；没有为 MF 单独选择另一学习率。

根目录的 `lr_sweep_selection.json` 保存选择结果。`reports` 还包含：

- `lr_k_heatmap.png` / `.svg`：lr × K 的验证 Recall/Hit/NDCG 热图。
- `lr_k_comparison.csv`：全部候选各自最佳轮次的结果。
- `lr_best_curves.png` / `.svg`：每个 lr 下最佳 K 的训练与验证曲线。
- `lr_*/`：每个 lr 的完整 K0–K4 和 MF 对比报告。

## 3. 搜索完成后做一次最终测试

```powershell
python .\week03-lightgcn\day15_experiments.py --stage final `
    --output-dir .\week03-lightgcn\experiments\day15_lr_sweep
```

读取冻结的 lr/K 选择，复用原 `run_day15.run_final`，仅测试选中 lr 下的 K0 和最佳 K。
数据、图和默认设备从搜索配置恢复。训练图仍只使用 Train，测试候选屏蔽 Train∪Valid。
测试结果保存在选中 `lr_*` 子目录的 `final_test_results.json`，随后更新图表。
一旦已有最终测试记录，该搜索目录不能继续 sweep 或重复 final，仍可用 plot 查看：

```powershell
python .\week03-lightgcn\day15_experiments.py --stage plot `
    --result-dir .\week03-lightgcn\experiments\day15_lr_sweep
```

原 `Test_Day15.py` 可对选中 lr 的子目录执行完整验收，将该子目录传给 `--result-dir`。

## 哪些属于消融实验

固定其他配置比较 K0–K4，是传播层数消融；K0 去掉了图传播，是同实现的 BPR-MF 基线。
原 Day 15 的五组实验已经在做这项消融，是否已运行完成应以实际实验记录为准。
多 lr 搜索属于超参数搜索。跨 lr 直接比较同时改变了学习率与层数的两个模型，
不能单独用来归因层数的影响，因此报告按每个 lr 分别提供层数消融与同 lr MF 对照。
这些实验没有单独消融 L2 正则、负采样方式或层融合方式，也不能仅凭层数曲线证明过平滑。
