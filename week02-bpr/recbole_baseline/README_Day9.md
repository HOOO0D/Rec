# Day 9：RecBole BPR 对照实验使用说明

该练习包承接 Day 7（SGD + D_S 均匀 Bootstrap）与 Day 8（Full Ranking）。
全部**自编写 Python 文件的注释和提示均使用中文**；RecBole 官方源码保留英文，不需要修改第三方代码。

## 文件清单

- `recbole_bpr.yaml`：RecBole 内置 BPR + ml-100k 基线配置（10 Epoch，Adam，全排序）。
- `run_day9.py`：可直接运行的 RecBole 启动器；保存模型、解析后配置、源码定位及评价结果。
- `protocol_audit.py`：需要完成 3 个 TODO；用于规范化、对比 Day 7 / RecBole 协议。
- `Test_Day9.py`：A–E 五组验收，100 分；只有真实运行完成才能拿满分。
- `Day9_notes.md`：论文与源码理解、结果表和 Week 2 复盘。

## 强烈建议独立环境（Windows PowerShell）

原来的 `torch-learning` 环境主要用于手写 PyTorch 模型，建议不要改动它。
RecBole 1.2.1 已有关于 NumPy 2 与 PyTorch >=2.6 的兼容性报告：
https://github.com/RUCAIBox/RecBole/issues/2215

```powershell
conda create -n recbole-day9 python=3.10 -y
conda activate recbole-day9
python -m pip install "numpy==1.26.4" "pandas<3" "torch==2.5.1" "recbole==1.2.1"
python -m pip check
python -c "import recbole, torch, numpy; print(torch.__version__, numpy.__version__)"
```

如果下载不畅，可沿用你已经配置的 pip 镜像。以 CPU 为主，无需额外安装 CUDA。不要直接在旧环境中强行降级 PyTorch。

## 放置方式

将本目录中的五个文件放到 `E:\AI\rec-learning\week02-bpr\`，不要覆盖 Day 7 的 `train.py` 或 Day 8 的 `evaluate.py`。

## 建议执行顺序

```powershell
cd E:\AI\rec-learning\week02-bpr

# 1. 完成 protocol_audit.py 的三个 TODO，先验收静态配置与协议比较。
python .\Test_Day9.py

# 2. 运行真正的 RecBole BPR 实验；RecBole 内部会自行使用其内置 ml-100k。
python .\run_day9.py --config .\recbole_bpr.yaml --output-dir .\runs\day9_recbole_adam

# 3. 对照 Day 7 的真实 SGD 实验配置，保存差异报告。
python .\protocol_audit.py --day7-run .\runs\day7_sgd_ml100k --recbole-run .\runs\day9_recbole_adam

# 4. 执行完整验收。
python .\Test_Day9.py --run-dir .\runs\day9_recbole_adam
```

A–C 共 60 分；D 检查你确实安装的 RecBole 源码（15 分）；E 检查真实训练、配置、模型和协议报告（25 分）。代码与笔记都通过，才可进入 Day 10。

## 输出目录

```text
runs/day9_recbole_adam/
├── recbole_bpr.yaml           # 运行时 YAML 快照
├── resolved_config.json      # 合并默认值后的真实配置
├── environment.json          # Python、Torch、NumPy 和 RecBole 版本
├── source_locations.json     # 本地源码位置和关键函数行号
├── run_result.json           # 最佳验证指标和框架自动得到的测试指标
├── protocol_comparison.json  # 与 Day 7 手写实验的协议差异
└── saved/                    # RecBole 官方模型 Checkpoint
```

**数据集提醒**：RecBole 自带 ml-100k Atomic Files，非 Day 7 的 `u.data` 原始零基数据读取器。RecBole 的 token 映射通常预留 padding ID 0，因此日志中的用户数和物品数可能是 944、1683，而原始数据的真实用户数/物品数是 943、1682。它们不是多出真实用户/电影。

**比较提醒**：框架的 `run_recbole()` 一次运行结束后会自动计算测试集结果。你只能记录该结果，不要根据测试结果反复调整学习率等设置。Day 7 的测试集尚未用于选配置，也不应借 Day 9 的结果对另一套协议进行排名。即使 K 相同、Full Ranking 相同、随机种子相同，数据划分、负采样、正则化、初始化和 ID 映射也可能不同。

**数据下载**：RecBole 官方文档说明 `ml-100k` 为自带示例数据；如果本地安装没有该数据，不要把 `u.data` 直接改名成 `.inter`，需按 Atomic File 规范转换。
