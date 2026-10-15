# Day2：Embedding 和批数据（270 分钟）

依据上传 RoadMap.md 的压缩版 Day2；Day1 的 A4/MF 不作为已掌握前提。
今天交付 embedding.py，实现其 TODO，并通过 Test_day2.py。
Day3 再整合评分模型、Loss 下降及保存重载；Day5 学 MF 原理。

## 章节、用时与具体验收

| 顺序 | 章节与范围 | 用时（阅读/代码） | 验收标准 |
|---|---|---:|---|
| A | PyTorch nn.Embedding：说明、num_embeddings、embedding_dim、weight、Shape、基础示例 | 50（20/30）分钟 | 能解释表是 [N,d] 的可训练参数；合法 ID 范围 0 到 N-1；本练习统一 long；[B] 查表得 [B,d]，重复 ID 得相同向量；通过 A |
| B | Learn the Basics → Datasets & DataLoaders：Creating a Custom Dataset for your files、__init__、__len__、__getitem__ | 45（15/30）分钟 | 独立实现三个接口；7 条数据返回长度7，每条为两个 long 标量及一个 float32 标量；空集长度0；通过 B |
| C | 同篇：Preparing your data for training with DataLoaders、Iterate through the DataLoader | 45（15/30）分钟 | 解释 batch_size、shuffle、drop_last；7条数据 batch_size=3 得 [3,3,1] 三批；三字段均为 [B]；打乱前后均无丢失/重复；通过 C |
| D | D2L 2.3 线性代数：2.3.6 点积、2.3.7 矩阵-向量积（后者只复习维度）；结合 Embedding 写成对点积 | 40（10/30）分钟 | 手算 [1,2]·[1,1]=3；两组 [B,d] 按最后维逐元素乘并求和，输出 [B]；B=1 保持 [1]；通过 D |
| E | D2L 2.5 自动微分：2.5.1 一个简单的例子；复习 PyTorch Optimization Loop 的清梯度/反传/更新 | 30（15/15）分钟 | 用查表结果求和作练习 loss，连续两次独立前向/反传/SGD 更新；访问行有梯度且更新，重复 ID 梯度累加，未访问行梯度0且不变；通过 E |
| 复盘 | 整合运行、口述并记录疑问 | 60（代码30/复盘30）分钟 | 全部通过；不看代码回答下述五题 |

总计阅读75分钟、代码165分钟、复盘30分钟。无须学完整第2/3章。
官方数据教程的 FashionMNIST/图像转换部分只浏览示例结构，不要求下载图片或学习视觉任务。
Embedding 的 padding_idx、max_norm、sparse 等高级选项今天跳过；E 使用默认 dense 梯度及无 momentum、无 weight_decay 的 SGD，未访问行不变的结论限于此配置。

## 资料

- Embedding：https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- Dataset/DataLoader：https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- D2L 2.3：https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html
- D2L 2.5：https://zh.d2l.ai/chapter_preliminaries/autograd.html
- Optimization Loop：https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

## 使用方法

在本目录、已安装 PyTorch 的环境中执行：

```powershell
python Test_day2.py A
python Test_day2.py B C
python Test_day2.py D E
python Test_day2.py
```

只补 embedding.py 中 TODO，不改验收断言。初始运行会报告未实现，这是正常状态。
全部五组 PASS + ALL DAY2 TESTS PASSED 才算代码通过；单组通过不算整体验收。
遇到 FAIL 根据题号定位，断言错误可直接调用对应 check_A 等函数查看完整堆栈。

## 口述验收（每题用自己的话说明，脚本不能代替）

1. 为什么用户 ID 只是类别索引，不能把 ID 的数值大小直接当作偏好强弱？为什么刚初始化的向量还没有学到偏好？
2. N=100、d=16、B=3 时，参数表、ID、查表输出、成对得分分别是什么 shape？
3. Dataset 和 DataLoader 各负责什么？shuffle 是否改变样本总量？为什么最后一批可能只有1条？
4. 成对点积 [B] 与所有两两组合 [B,B] 有什么区别？为什么 sum(dim=...) 的轴很重要？
5. Embedding.weight 为什么能被优化器更新？ID 自身需要梯度吗？同一个 ID 出现两次为何会累加梯度？

今天不以 loss 下降为硬门槛：E 的求和 loss 专门验证梯度和更新机制，不是推荐系统训练目标。
