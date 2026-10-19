# Week 1 学习复盘（待填写）

## 1. 张量和评分形状
TODO：解释广播、点积与矩阵乘法；写出 ID、参数表、查表输出、成对得分的 shape。

## 2. Embedding 和模型
TODO：解释 ID 为何不需要梯度、weight 为何是可训练参数、__init__/forward 的职责。

## 3. 训练闭环
TODO：不看教程写出 zero_grad/forward/loss/backward/step 顺序；解释梯度累积及每批重新前向。

## 4. 真实运行记录
TODO：记录 Python/PyTorch 版本、种子42、CPU、embedding_dim=2、SGD lr=0.05、batch_size=4、400 epochs。
TODO：记录初始/最终 MSE、至少一张参数表更新前后的数值，以及 A-F 验收结果；不要填写未实际运行的结果。

## 5. 保存与重载
TODO：解释 state_dict、相同模型结构、eval() 与 no_grad() 的区别；记录重载前后最大预测误差。

## 6. 结论与遗留问题
TODO：解释为何微型训练数据 loss 下降不代表真实推荐效果；记录尚未理解的问题。
