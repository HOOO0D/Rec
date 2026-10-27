# Day 9｜RecBole BPR 源码阅读、协议审计与 Week 2 复盘

> 今日目标：用 RecBole 跑通 BPR + ml-100k，并判断与 Day 7–8 手写实现的可比性。绝不根据测试集反复挑选超参数。

## 一、阅读范围与时间记录

- RecBole General Recommendation Quick Start：
- RecBole `run_recbole()` 工作流程：
- `BPR.__init__` / `calculate_loss` / `full_sort_predict`：
- `BPRLoss.forward`：
- 实际投入时间：

## 二、必须不看源码回答的五个问题

1. `BPR.calculate_loss(interaction)` 从输入中读取哪些字段？三种 ID 的 shape 分别是什么？
2. `BPR.forward()` 的返回值是什么？是否直接返回一维评分张量？
3. `BPRLoss.forward()` 具体如何计算正负评分差？为什么实现中会出现一个小常数？这与稳定的 `softplus` 有何数值区别？
4. RecBole 的标准 BPR 模型是否直接在 `calculate_loss()` 中加入 L2？如果需要正则化，框架的优化器与模型代码分别负责什么？
5. `full_sort_predict()` 与 `predict()` 在输出维度、候选集合上有什么区别？哪一层处理训练已交互物品过滤？

## 三、RecBole 实验记录（从运行产物填写）

- Python / RecBole / PyTorch / NumPy 版本：
- 配置 YAML 路径：
- resolved_config.json 路径：
- 模型文件路径：
- 优化器 / 学习率 / Batch Size / Embedding Dim：
- 正则化权重：
- 训练负采样的策略：
- 数据集来源（框架内置 ml-100k / 自行构建）：
- 用户 / 物品数量（RecBole 可能保留 0 索引作 Padding）：
- 数据划分、排序候选集、过滤规则、评价 K：
- best_valid_result（只基于验证集选择配置）：
- test_result（框架自动给出，仅记录，不用于调参）：

## 四、与 Day 7–8 的协议差异（必须逐项填写）

| 对比维度 | Day 7–8 手写实现 | RecBole 本次实验 | 是否严格一致 | 解释 |
|---|---|---|---|---|
| 双 Embedding 点积 |  |  |  |  |
| Embedding 初始化 |  |  |  |  |
| Optimizer |  |  |  |  |
| 负采样：用户选择 / 物品选择 / 有放回性 |  |  |  |  |
| 正则项定义与归一化 |  |  |  |  |
| 数据集来源与用户物品映射 |  |  |  |  |
| Train/Valid/Test 实际交互划分 |  |  |  |  |
| 评价候选全集与屏蔽规则 |  |  |  |  |
| Recall、Hit、NDCG 计算与宏平均 |  |  |  |  |
| 最佳模型选择与测试次数 |  |  |  |  |

请明确指出：即使 seed 相同，两套不同的数据划分程序仍可能得到不同的实际交互集合。不同协议的绝对指标不能当成公平的模型优劣比较。

## 五、Day 9 自动验收及 Week 2 收尾

- [ ] `Test_Day9.py` 100/100，包含真实 RecBole 运行及协议报告
- [ ] `source_locations.json` 中定位 BPR 源码，并完成上面 5 道题
- [ ] 能清楚解释 RecBole BPR 的计算流程与手写 BPR-MF 的联系
- [ ] 已保存 YAML、解析配置、Python 环境、日志、模型与验证/测试结果
- [ ] 已分析协议差异，拒绝在不同协议下直接比较指标
- [ ] 能完整口述 Week 2 的数据 → 三元组 → 模型 → BPR Loss → SGD → Top-K 评价

### Week 2 复盘
- 能脱离教程独立实现的代码：
- 最难理解的三个概念：
- 当前 BPR-MF 的可靠性与局限：
- 今后 LightGCN 将复用的训练与评价模块：
- 进入 Day 10 图论与用户—物品二部图之前需要补练的内容：
