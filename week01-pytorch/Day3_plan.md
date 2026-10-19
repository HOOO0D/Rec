# Day3：独立整合可训练的用户—物品评分模型

按当前 RoadMap 的压缩版 Week1 Day3；Day2 已完成（本人反馈）。今日目标：
两个 Embedding -> 成对点积 -> MSE -> 反传/更新 -> 评估 -> 保存/重载。
此模型与 MF 的点积参数化一致，今天重点验证 PyTorch 闭环，MF 的系统理论仍放 Day5。

## 学习章节及具体标准（270分钟）

| 阶段 | 章节与阅读范围 | 阅读/代码/分析/复盘（分钟） | 具体验收 |
|---|---|---|---|
| A 批数据复习 | PyTorch Datasets & DataLoaders 的批处理和遍历部分 | 5/10/0/0 | 7条数据 batch_size=3 得 [3,3,1]；两个 long ID、一个 float32 rating；无丢失，通过 A |
| B 模型整合 | Build the Neural Network：Define the Class、Model Parameters；复习 Embedding | 15/25/5/0 | 独立写 __init__/forward；两张表注册为参数；ID [B] -> 向量 [B,d] -> 分数 [B]；B=1 保留维度；手算一例，通过 B |
| C 损失 | D2L 3.1.1 线性回归的基本元素中的损失函数、3.2.5 定义损失函数；PyTorch Optimization 的 Loss Function | 15/10/5/0 | 理解 MSE=mean((预测-评分)^2)；[1,3] 对 [2,1] 的 MSE=2.5、预测梯度=[-1,2]；输出可求导标量，通过 C |
| D 训练 | PyTorch Optimizing Model Parameters：Hyperparameters、Optimizer、Full Implementation 中 train_loop；D2L 3.2.6/3.2.7 | 20/40/10/0 | 不照抄写完整每批训练顺序；每批清梯度、重新前向；所有批都更新；epoch loss 按样本数加权；两表均有有限梯度且更新，通过 D |
| E 评估与排错 | 同教程 test_loop，替换分类准确率为评分 MSE | 10/20/15/0 | eval + no_grad，不修改参数或梯度；固定微型数据400轮后 MSE<0.02 且低于初始10%；不要求每轮单调下降，通过 E |
| F 保存重载 | Save and Load the Model：Saving and Loading Model Weights | 10/15/10/0 | 只保存 state_dict；新建同结构模型并加载；相同输入预测最大绝对差不超过1e-7；通过 F |
| G 周验收笔记 | week01_notes.md、RoadMap 周验收五项 | 0/0/0/30 | 填完所有 TODO；记录真实实验结果；口述 shape/广播/点积/矩阵乘法、训练顺序、Module、Embedding 和 Loss/梯度/更新 |

合计：阅读75、代码120、分析45、复盘30分钟。不能在4–5小时内完成则补练，不跳过验收。

## 章节链接

- 模型：https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- 数据：https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- 优化：https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- 保存重载：https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- D2L 3.1：https://zh.d2l.ai/chapter_linear-networks/linear-regression.html
- D2L 3.2：https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html

不必重学整章或照搬图像分类网络；今天不用 CNN、Softmax 分类、BPR、负采样、Top-K 或真实数据下载。
资料中的分类交叉熵改用 MSE；不能把预测强制取整或用 .item() 后再求导。

## 文件与运行

- simple_recommender.py：补齐 TODO A-F；数据使用 TensorDataset，自包含，不依赖 Day2 的实现。
- Test_day3.py：参考 Test.py 的 assert/[PASS] 风格，不改断言；自动提供微型数据。
- week01_notes.md：填入实际理解及结果；自动代码测试不代替人工笔记与口述验收。

在本目录激活已安装 PyTorch 的环境后：

```powershell
python Test_day3.py A B C
python Test_day3.py D
python Test_day3.py E F
python Test_day3.py
```

未补 TODO 时 FAIL 是正常状态。全部 A-F PASS + ALL DAY3 TESTS PASSED 才算代码通过。
失败后可以在 Python 中直接调用对应 check_D() 等函数查看完整堆栈。
Test_day3 的 checkpoint 保存在临时目录，结束自动清理；如需长期保存，可手动调用 save_weights。

训练数据是3个用户、4个物品的12条完整合成评分，只用于检查能否拟合。
E 的前后 MSE 使用同一模型在整份数据上评估，避免拿不同 batch 的 loss 直接比较。
train_one_epoch 返回的是每个 batch 更新前的误差汇总，与整轮结束后的 evaluate_mse 不一定相同。
今天只保存推理权重；恢复训练的优化器状态等留待后续。
Week1 最终门槛：所有代码验收通过，笔记完成，并能不看教程重新写核心训练循环。
通过后按 RoadMap 安排1天休息/机动，再进入 Day4 推荐基本概念。
