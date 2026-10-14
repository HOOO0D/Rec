# 推荐系统科研入门：六周学习计划

> **学习周期**：保留六阶段（Week 1–6）主题；Week 1 压缩为 3 个学习日，其余各 6 日，共 33 个学习日（含已完成 1 日，剩余 32 日）  
> **每日投入**：4–5 小时；建议以 4.5 小时安排  
> **总投入**：剩余计划约 128–160 小时；已完成部分以实际用时记录，不按原四天重复计入  
> **当前起点**：已有编程基础和部分推荐系统概念；截至 2026-10-14 已掌握原 Day 1–4 的 PyTorch 基础  
> **终点**：完成 BPR、LightGCN、SGL、DirectAU、SASRec 五篇论文的第一轮学习，能够实现基本算法、运行标准基线、分析实验结果  
> **更新日期**：2026-10-14

## 0. 先看这三条原则

1. **原理和代码并行**：每天至少 2 小时编程；不以“看完视频”为验收标准。
2. **不同论文的实践深度不同**：BPR-MF 和 LightGCN 尽可能独立实现；SGL 在 LightGCN 上实现核心增量；DirectAU 重点实现 Loss；SASRec 重点读懂 Attention 并在 RecBole 中运行、修改。
3. **先正确，再涨指标**：评价协议、数据划分、防止数据泄漏、固定随机种子等优先于参数技巧。六周不要求严格复现论文所有表格，也不保证得到论文原始指标。

### 每日时间安排（以 270 分钟为例）

| 模块 | 时间 | 做什么 |
|---|---:|---|
| 理论学习 | 75 分钟 | 视频、教材、论文阅读、公式理解 |
| 编程实践 | 120 分钟 | 独立写代码、运行测试、排查错误 |
| 深入分析 | 45 分钟 | 对照源码、画结构图、阅读实验结果 |
| 整理复盘 | 30 分钟 | 记笔记、记录问题、提交 Git |

> Week 表示学习阶段，不再严格对应自然周。Week 1 在 Day 3 验收后安排 1 天休息/机动；后续每完成 6 个学习日安排 1 天休息/机动，Day 编号不包含休息日。若全部通过，较原计划提前 3 个学习日，约 5.5–6 个自然周完成；不固定后续完成日期。每日仍以 4–5 小时为上限，未通过验收则补练，不把欠账带到下一阶段。


### 本次压缩规则与依赖门槛

- 新 Day 1 对应原 Day 1–4；新 Day 2–33 分别对应原 Day 5–36。只压缩已掌握的基础，不删除后续任务；原目录名继续保留。
- 下一步是 Day 2 的 Embedding / Dataset / DataLoader，随后 Day 3 独立整合评分模型。通过 Week 1 验收后再进入 BPR。
- Week 2（Day 4–9）保留负采样与评估；Week 3（Day 10–15）复用其训练和评估进入 LightGCN；Week 4（Day 16–21）在 LightGCN 上增量实现 SGL。
- Week 5（Day 22–27）保留 DirectAU 和 Attention；Week 6（Day 28–33）必须在 Attention 与时序数据协议准备好后进入 SASRec。
- 后续默认不再压缩。只有能独立解释原理、写核心代码、完成对应验收，才允许提前进入下一学习日；训练等待可用于读论文或整理日志，但不能跳过结果分析。Day 15 的 BPR/LightGCN 里程碑仍是硬门槛。

## 1. 六阶段总览（33 个学习日）

| 周次 | 主线 | 学习成果 / 交付物 | 关键验收 |
|---|---|---|---|
| 第 1 周（Day 1–3） | 已掌握基础 → Embedding → 训练整合 | 可训练的简易评分模型 | 不看教程独立写训练循环 |
| 第 2 周 | 推荐系统基础 + BPR-MF | `bpr/`、负采样、Top-K 评估脚本 | 从 `(u,i,j)` 训练 Embedding 并评估 |
| 第 3 周 | 图论/GNN 基础 + LightGCN | `lightgcn/`、图传播及层数对比 | 独立实现图传播，与 BPR-MF 对比 |
| 第 4 周 | SGL + 对比学习 | 图增强、InfoNCE、消融实验 | 实现 SGL-ED 简化版 |
| 第 5 周 | DirectAU + Attention 预备 | Alignment / Uniformity Loss、Attention 小练习 | 理解表示坍缩并能手算注意力 |
| 第 6 周 | SASRec + 实验总结 | SASRec 训练记录、六周总结报告 | 跑通序列推荐并解释模型工作流程 |

## 2. 第 1 周：PyTorch 基础收尾与整合（Day 1–3）

**周目标**：理解 PyTorch 最小训练闭环，使用 `nn.Embedding` 编写简易推荐评分模型。

**主要资料**：

- [《动手学深度学习》（中文版）第 2 章：预备知识](https://zh.d2l.ai/chapter_preliminaries/index.html)
- [《动手学深度学习》第 3 章：线性神经网络](https://zh.d2l.ai/chapter_linear-networks/index.html)
- [PyTorch 官方 Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch `nn.Embedding` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

### 逐日清单

- [x] **Day 1｜已完成基础（2026-10-14；原始 Day 1 至 Day 4）**：已掌握 Tensor、矩阵运算、自动求导、`nn.Module` 和 Optimizer；原任务文件为 `tensor_basics.py`、`tensor_ops.py`、`autograd.py`、`linear_regression.py`。完成状态依据本人反馈，代码交付与独立验收仍需核对；不重复安排四天课程。
  - **当日验收**：不看教程写出小型 nn.Module 和完整训练循环；手算并核对一次矩阵乘与多项式梯度；解释梯度累积，并用参数前后差异确认 step 有效。若仍需照抄，Day 2 开始前针对薄弱点补练。
- [ ] **Day 2｜Embedding 和批数据**：学习 `nn.Embedding`、`Dataset`、`DataLoader`、Batch 维度；从用户/物品 ID 得到向量并计算点积；完成 `embedding.py`。
  - **当日验收**：提交 embedding.py；一个批次的 ID 为 [B]、向量为 [B,d]、点积得分为 [B]；ID 使用整数类型；反传后被访问的 Embedding 行有梯度且更新，DataLoader 能遍历完整数据。
- [ ] **Day 3｜周综合练习**：用两个 Embedding 表和一个简单 Loss 训练“用户—物品评分”小模型；检查参数确实发生更新；完成 `simple_recommender.py` 和 `week01_notes.md`。
  - **当日验收**：不照抄教程完成 simple_recommender.py；在固定种子的微型评分数据上观察 Loss 下降和参数更新；保存并重载后同一输入预测一致；提交 week01_notes.md 并通过本周全部验收。

### 周验收

- [ ] 不看教程解释 `Tensor.shape`、广播、向量点积、矩阵乘法。
- [ ] 能说出 `zero_grad()`、`backward()`、`step()` 的作用与顺序。
- [ ] 能独立实现 `nn.Module` 的 `__init__` 和 `forward()`。
- [ ] 能解释 Embedding 表为什么是可训练参数、用户 ID 为什么能映射成一个向量。
- [ ] 能训练至少一个小模型，确认 Loss、梯度、参数更新过程合理。

> **第 1 周无需学习**：CNN、计算机视觉、复杂 RNN、分布式训练、多卡训练、PyTorch 底层 CUDA 编程。

## 3. 第 2 周：协同过滤 + BPR（Day 4–9）

**周目标**：理解隐式反馈排序，独立实现 BPR-MF 并做基本 Top-K 推荐评价。

**主要资料**：

- [《动手学深度学习》推荐系统第 21 章（英文官方版）](https://www.d2l.ai/chapter_recommender-systems/index.html)：重点 21.1、21.2、21.3、21.5
- [矩阵分解（21.3）](https://www.d2l.ai/chapter_recommender-systems/mf.html)
- [个性化排序（21.5）](https://www.d2l.ai/chapter_recommender-systems/ranking.html)
- [RecBole 通用推荐 Quick Start](https://recbole.io/docs/get_started/started/general.html)
- 上传的论文：**BPR: Bayesian Personalized Ranking from Implicit Feedback**

### 逐日清单

- [ ] **Day 4｜推荐基本概念**：复习召回、排序、UserCF、ItemCF、MF；明确显式/隐式反馈、已观察/未观察交互；写一页概念笔记。
  - **当日验收**：提交一页概念笔记，说明召回/排序、显式/隐式反馈、UserCF/ItemCF/MF；给出未交互不等于不喜欢的例子。
- [ ] **Day 5｜矩阵分解**：理解用户/物品 Embedding、点积评分、正则化；使用 PyTorch 写 MF 评分函数；手动检查 `shape`。
  - **当日验收**：MF 支持批量用户/物品 ID；手算一组点积并与程序核对；说明正则化对象和评分输出形状。
- [ ] **Day 6｜精读 BPR**：阅读论文第 1、3、4 节；理解训练三元组 `(u, i, j)`、BPR Loss、负采样；在纸上计算样例 Loss。
  - **当日验收**：写清 (u,i,j) 的约束；手算得分差为 0 时 Loss 为 log(2)，并验证正物品得分提高时 Loss 降低；记录论文假设。
- [ ] **Day 7｜独立实现 BPR-MF**：编写模型、负采样器、训练循环；先在合成数据上验证，再在 MovieLens-100K 上训练；保存日志。
  - **当日验收**：负采样不取用户训练正物品，并处理无可采负例的用户；合成数据能学到正分高于负分；ml-100k 跑通一轮并保存划分、种子、配置与日志。
- [ ] **Day 8｜评价指标**：学习 Recall@K、Hit@K、NDCG@K；实现 Top-K 评估并过滤训练集中已交互物品；保持验证/测试集独立。
  - **当日验收**：用手工排名验证 Recall、Hit、NDCG；过滤训练已交互物品，检查三份划分无交互重叠；固定候选集和 K，输出验证指标。
- [ ] **Day 9｜RecBole 对照**：用 RecBole 运行 `BPR + ml-100k`；阅读内置 BPR 的 `calculate_loss()`；比较代码结构并写第一份实验记录。
  - **当日验收**：RecBole BPR 跑通并保存配置；读懂 calculate_loss；逐项核对划分、候选集、正则化和采样差异，协议不同则不作指标优劣结论。

### 核心公式

$$
\hat y_{ui}=\mathbf e_u^\top\mathbf e_i
$$

$$
\mathcal L_{\mathrm{BPR}}=-\log\sigma(\hat y_{ui}-\hat y_{uj})
$$

### 周验收

- [ ] 不借助 RecBole，独立完成 BPR-MF 模型、训练及预测。
- [ ] 知道为什么“未交互”不代表真实负反馈，理解负采样的局限。
- [ ] 能对给定 Embedding 手算正负物品得分和 BPR Loss。
- [ ] 能输出用户 Top-10 推荐，以及 Recall@K / NDCG@K 等指标。
- [ ] 能解释训练 Loss、验证集指标和测试集指标的区别。

## 4. 第 3 周：LightGCN（Day 10–15）

**周目标**：学会用户—物品二部图构造、归一化邻居聚合和多层表示组合。

**主要资料**：

- [Stanford CS224W 课程主页](https://web.stanford.edu/class/cs224w/)：只选图表示、GNN、消息传递相关的公开讲义；不需要学完整门课。
- [RecBole LightGCN 文档](https://recbole.io/docs/user_guide/model/general/lightgcn.html)
- 上传的论文：**LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**

### 逐日清单

- [ ] **Day 10｜图论与二部图**：理解节点、边、邻居、节点度、邻接矩阵；把用户—物品交互表转成二部图。
  - **当日验收**：用至少 2 个用户和 3 个物品画二部图；核对节点编号、双向边、邻接矩阵形状和节点度，图仅由训练交互构造。
- [ ] **Day 11｜邻居聚合**：学习度矩阵 `D`、对称归一化邻接矩阵 `D^{-1/2} A D^{-1/2}`、消息传递、稀疏矩阵乘法。
  - **当日验收**：手算小图的一次对称归一化聚合，与稀疏乘法结果核对；对孤立节点不产生 NaN/Inf。
- [ ] **Day 12｜精读 LightGCN**：学习论文第 3 节方法与消融实验；解释为何去掉特征变换和非线性激活，理解 0 层与多层表示。
  - **当日验收**：写出 LightGCN 与普通 GCN 的差异，以及 0 层、多层融合和 BPR 目标；结合论文消融说明结论适用范围。
- [ ] **Day 13｜图传播实现**：使用 PyTorch 实现 `E^(k+1) = A_norm @ E^(k)` 和不同层的 Embedding 融合；先用 2–3 个用户的小图手算验证。
  - **当日验收**：手算小图的 1–2 层传播并与代码比较；输出形状一致、梯度能回传到初始 Embedding；0 层退化为 MF 评分。
- [ ] **Day 14｜训练 LightGCN**：连接 BPR Loss、优化器和评估器，构建可运行的简化版 LightGCN。
  - **当日验收**：LightGCN 接入已验证的 BPR 训练和评估；小数据可过拟合、梯度有限、验证指标可输出；邻接矩阵未使用验证或测试边。
- [ ] **Day 15｜基线对比与消融**：运行 BPR-MF、LightGCN；固定数据划分和评估协议，比较 `n_layers = 0,1,2,3,4`；整理结果。
  - **当日验收**：固定协议比较 BPR-MF 与 LightGCN 的 0–4 层结果；记录种子、配置、验证选择与最终测试结果；能独立解释两者全流程才进入 SGL。

### 核心公式

$$
\widetilde A=D^{-1/2} A D^{-1/2},\qquad
E^{(k+1)}=\widetilde A E^{(k)}
$$

$$
E=\sum_{k=0}^{K}\alpha_k E^{(k)}
$$

### 周验收

- [ ] 能画出用户—物品二部图，解释第 1、2 层图传播引入的不同信息。
- [ ] 独立实现 `torch.sparse.mm` 参与的多层传播。
- [ ] 能计算用户/物品最终 Embedding，并用 BPR Loss 训练。
- [ ] 能分析层数变化与指标变化，不把“效果下降”直接归因为单一因素。

> **里程碑 A（第 3 周末）**：如果 BPR-MF 和 LightGCN 的原理、核心代码、训练和评价都已经打通，就已具备进入基础推荐算法复现任务的技术准备。

## 5. 第 4 周：SGL 与对比学习（Day 16–21）

**周目标**：在 LightGCN 上加入图增强及 InfoNCE 对比损失，理解自监督辅助训练。

**主要资料**：

- [RecBole SGL 文档](https://recbole.io/docs/user_guide/model/general/sgl.html)
- 上传的论文：**Self-supervised Graph Learning for Recommendation**

### 逐日清单

- [ ] **Day 16｜自监督与对比学习**：理解自监督任务、正负样本对、两种数据视图、对比学习的监督信号。
  - **当日验收**：画出同一节点两视图的正对及其他节点的负对；解释监督信号来源，列出与 BPR 负物品的区别。
- [ ] **Day 17｜InfoNCE**：补充余弦相似度、Softmax、温度参数 `tau`、Hard Negative；手算一个小批次的对比损失。
  - **当日验收**：手算小批次余弦相似度和 InfoNCE，与程序核对；改变 tau 比较分布；使用稳定计算并确认损失和梯度有限。
- [ ] **Day 18｜精读 SGL**：学习图增强的 Node Dropout / Edge Dropout / Random Walk，以及主任务与自监督任务关系。
  - **当日验收**：整理三种图增强及主任务/辅助任务关系；明确本次实现 SGL-ED 的范围和与论文实现的差异。
- [ ] **Day 19｜图增强代码**：在 LightGCN 上实现 Edge Dropout，构造两个独立增强图视图；检查保留边比例。
  - **当日验收**：两视图独立采样、节点编号一致；无向边成对保留/删除；记录实际保留比例，drop_ratio=0 时恢复原图并重新计算归一化。
- [ ] **Day 20｜联合训练**：实现 InfoNCE Loss，联立 BPR Loss 和自监督 Loss，完成 SGL-ED 简化版。
  - **当日验收**：联合 Loss 可反传至 Embedding；分别记录 BPR/SSL/总 Loss；ssl_weight=0 时与相同配置 LightGCN 训练目标一致。
- [ ] **Day 21｜对照与消融**：用 RecBole SGL 对照；测试不同 `ssl_weight` 或 `drop_ratio`，注意逐次只控制一个变量。
  - **当日验收**：跑通 RecBole SGL 对照；一次仅改变 ssl_weight 或 drop_ratio，至少基线加一个变体；保存协议差异、结果表和原因分析。

### 周验收

- [ ] 能清楚区分 BPR 中的负物品与 InfoNCE 中的对比负样本。
- [ ] 能解释同一个节点两个增强表示为何构成正样本对。
- [ ] 能解释温度参数对分布和难负样本权重的影响。
- [ ] 简化版 SGL-ED 能训练、评估，并有至少一个消融结果。

## 6. 第 5 周：DirectAU 与 Attention（Day 22–27）

**周目标**：实现 DirectAU 的两个损失项，理解表示坍缩；随后补齐 SASRec 所需的注意力基础。

**主要资料**：

- [DirectAU 论文作者仓库](https://github.com/THUwangcy/DirectAU)
- [《动手学深度学习》第 10 章：注意力机制](https://zh.d2l.ai/chapter_attention-mechanisms/index.html)
- 上传的论文：**Towards Representation Alignment and Uniformity in Collaborative Filtering**

### 逐日清单

- [ ] **Day 22｜表示空间基础**：学习 L2 归一化、向量距离、Alignment、Uniformity；解释为什么只追求正样本靠近会有表示坍缩风险。
  - **当日验收**：对一组向量手算 L2 归一化、距离；用全部向量相同的例子说明 Alignment 好但 Uniformity 差，记录坍缩风险。
- [ ] **Day 23｜精读 DirectAU**：理解论文理论动机与第 4 节优化目标；和 BPR、SGL 进行比较。
  - **当日验收**：用自己的话写清 DirectAU 目标、正交互取样和两项损失的作用；列出与 BPR、SGL 的区别及论文局限。
- [ ] **Day 24｜实现 DirectAU**：使用 PyTorch 编写 Alignment Loss、Uniformity Loss，在 MF Encoder 上训练。
  - **当日验收**：独立实现两项 Loss 并与手工小例核对；处理批次不足两个有效向量的情形，损失/梯度有限；MF Encoder 可完成训练。
- [ ] **Day 25｜对照实验**：对照论文作者源码，在同一数据划分和评估协议下比较 BPR 与 DirectAU；记录 `gamma` 影响。
  - **当日验收**：核对作者源码的归一化、距离及权重定义；固定 CF 协议比较 BPR 与 DirectAU，至少完成一组 gamma 对照并保存结果。
- [ ] **Day 26｜Attention 基础**：学习 Query/Key/Value、缩放点积注意力和 Softmax；手算一个极小 Attention 例子。
  - **当日验收**：写明 Q/K/V、分数与输出形状；手算极小注意力权重并与程序核对，各行权重和为 1。
- [ ] **Day 27｜Self-Attention 基础**：学习 Multi-Head Attention、位置编码、Causal Mask、FFN、残差、LayerNorm；实现简单的因果注意力。
  - **当日验收**：实现因果注意力并解释多头、位置、FFN、残差和 LayerNorm；在 eval 模式改变未来输入，前缀输出保持一致；输出无 NaN。

### 核心公式

$$
\mathcal L_{\mathrm{DirectAU}}=\mathcal L_{\mathrm{align}}+\gamma\mathcal L_{\mathrm{uniform}}
$$

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### 周验收

- [ ] 可以解释 Alignment、Uniformity 分别优化什么。
- [ ] 能自己实现并运行 DirectAU Loss，而不只是调用作者仓库。
- [ ] 能写出 QKV 的张量形状，并手算一个 Attention 权重矩阵。
- [ ] 知道 Causal Mask 为什么用于预测下一物品。

## 7. 第 6 周：SASRec 与最终总结（Day 28–33）

**周目标**：读懂 SASRec 并开展初步序列推荐实验，形成可复查的学习成果。

**主要资料**：

- [RecBole SASRec 文档](https://recbole.io/docs/user_guide/model/sequential/sasrec.html)
- [RecBole 序列推荐 Quick Start](https://recbole.io/docs/get_started/started/sequential.html)
- 上传的论文：**Self-Attentive Sequential Recommendation**

### 逐日清单

- [ ] **Day 28｜序列推荐基础**：学习行为顺序、下一物品预测、时间切分、行为序列截断/填充，理解与一般 CF 的区别。
  - **当日验收**：构造一个有时间戳的用户样例，列出输入/目标、截断、填充和时序划分；训练样本不读取验证/测试未来行为。
- [ ] **Day 29｜精读 SASRec**：研究 Item Embedding、Position Embedding、Self-Attention、FFN、预测层和训练目标；画结构图。
  - **当日验收**：画 SASRec 结构图并标注形状、位置和掩码；说明训练目标、预测候选及 RecBole 与论文可能存在的实现差异。
- [ ] **Day 30｜实现核心模块**：手写简化版因果 Self-Attention，用单元测试确保不会读取未来位置的物品。
  - **当日验收**：完成简化因果 Self-Attention；测试改变未来物品不影响前缀输出，检查 padding 屏蔽及全 padding 边界不产生 NaN。
- [ ] **Day 31｜运行标准实现**：用 RecBole 在 MovieLens-100K 上训练 SASRec，确保加载时间戳并采用正确的时序协议。
  - **当日验收**：RecBole SASRec 跑通训练和评估；确认时间戳被加载、序列按时间构造；保存配置、切分、候选集、日志和指标。
- [ ] **Day 32｜参数研究**：固定数据与评估协议，比较序列长度/注意力层数等对性能的影响；至少完成一组消融。
  - **当日验收**：固定时序协议，一次只改变序列长度或层数，至少一组对照；验证集选配置，测试集最终评价，不与随机切分 CF 指标直接比较。
- [ ] **Day 33｜总结与整理**：补齐五篇论文的“问题—方法—公式—实验—局限”；整理代码仓库、结果表及待解决问题。
  - **当日验收**：五篇论文各有问题/方法/公式/实验/局限笔记；代码、配置、日志路径、结果表可追溯；逐项核对最终 10 项能力验收并列出未通过项和补练计划。

### 特别注意：SASRec 评估不同于一般协同过滤

- SASRec 是下一物品预测，通常需要按时间顺序构造输入和目标，并进行时序式评估。
- 不要直接把普通 CF 的随机交互划分结果与 SASRec 的时间留出结果做横向模型优劣比较。
- RecBole SASRec 默认实现与原论文在训练目标等细节上可能不同；初学先跑通，严格复现时再逐项核对。
- 即使使用相同的 MovieLens-100K 数据，候选集、切分方式、负采样和评价方式不一致时，指标也不可直接比较。

### 周验收

- [ ] 能不看论文讲清 SASRec 的数据输入、模型结构、训练与预测步骤。
- [ ] 能解释 QKV、位置编码、Causal Mask、FFN、残差连接的作用。
- [ ] 能用 RecBole 完成一次序列推荐训练和评估。
- [ ] 完成至少一份带实验设置和原因分析的总结报告。

## 8. 六阶段最终能力验收（10 项）

- [ ] **PyTorch**：不看教程定义 `nn.Module`，完成训练循环。
- [ ] **BPR-MF**：实现训练三元组、负采样、BPR Loss 和 Embedding 参数学习。
- [ ] **评价指标**：正确实现 Recall@K、NDCG@K，并解释不同评估协议的限制。
- [ ] **LightGCN**：能构造归一化邻接矩阵，实现多层图传播。
- [ ] **RecBole**：会通过配置运行 BPR、LightGCN、SGL 和 SASRec，理解主要接口。
- [ ] **SGL**：能实现 Edge Dropout 和简化 InfoNCE，并进行联合训练。
- [ ] **DirectAU**：能实现 Alignment/Uniformity Loss 并解释表示坍缩。
- [ ] **SASRec**：能解释 QKV、Causal Mask 和下一物品预测，并跑通模型。
- [ ] **科研实验**：至少做过一次公平的基线对比和一次控制变量消融。
- [ ] **论文表达**：能脱离原文介绍五篇论文各自解决的问题、方法、创新和限制。

**优先级最高的第二个里程碑：第 3 周结束时，保证独立掌握 BPR-MF 与 LightGCN 的“原理—实现—训练—评价”闭环。** 如果没有达到，宁可延长 1 周，也不要仅为了赶进度继续读后面的论文。

## 9. 预期项目目录

```text
rec-learning/
├── week01-pytorch/
│   ├── tensor_basics.py
│   ├── tensor_ops.py
│   ├── autograd.py
│   ├── linear_regression.py
│   ├── embedding.py
│   └── simple_recommender.py
├── week02-bpr/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── notes.md
├── week03-lightgcn/
│   ├── model.py
│   ├── train.py
│   └── experiments/
├── week04-sgl/
│   ├── augmentation.py
│   ├── contrastive_loss.py
│   └── experiments/
├── week05-directau/
│   ├── losses.py
│   └── experiments/
├── week06-sasrec/
│   ├── attention.py
│   └── experiments/
├── configs/
├── results/
└── README.md
```

建议在每周末提交 Git，并把训练配置、随机种子、结果、日志位置写进该周 `notes.md`。不建议把体积庞大的中间数据或模型权重直接提交到 Git 仓库。

## 10. 每次实验必须记录的最小信息

以下模板可直接复制到每周的 `notes.md`：

```markdown
### 实验记录：YYYY-MM-DD
- 任务与假设：
- 模型 / 对照模型：
- 数据集及版本：
- 数据过滤与训练/验证/测试划分：
- 全排序 / 采样排序与候选集：
- 已交互物品过滤方式：
- Embedding 维度、层数、学习率、Batch Size、正则化：
- 负采样方式：
- 随机种子及重复次数：
- 运行环境：PyTorch / RecBole / CUDA 版本：
- Recall@K、NDCG@K、Hit@K（验证集/测试集）：
- 日志及 Checkpoint 路径：
- 结果分析与下一步：
```

实验规范底线：

- **先固定协议再比较模型**，不能只对齐数据集名称。
- **验证集调参，测试集最终评价**；不要根据测试集指标反复选配置。
- **保存训练配置与随机种子**；条件允许时使用多个种子并报告均值与波动。
- **模型没涨指标不代表实现错误**；先排查实现和协议，再分析优化、模型归纳偏置、数据特性等原因。

## 11. 学习中需要暂时忽略的内容

| 内容 | 当前建议 |
|---|---|
| 冷启动 | 了解新用户、新物品冷启动的概念，不必现在专门复现论文 |
| 工业特征工程 / 线上涨指标技巧 | 暂缓，先掌握规范的离线实验和核心算法 |
| 大规模部署、分布式训练、多 GPU | 暂缓，入门数据集用 CPU 或单 GPU 即可 |
| 完整学习 CS224W | 没必要；只选择图表示与消息传递部分 |
| 全量学习 Transformer / 大语言模型 | 没必要；SASRec 阶段只补必要的 Self-Attention 模块 |
| 五篇论文所有证明、全部实验表格 | 第一轮不强求；保证核心公式、模型机制与重要结果理解即可 |

## 12. 官方参考资源汇总

| 资源 | 用途 | 链接 |
|---|---|---|
| 《动手学深度学习》中文 | PyTorch 与数学基础 | https://zh.d2l.ai/ |
| PyTorch Learn the Basics | Tensor、模型、Autograd、Optimizer | https://docs.pytorch.org/tutorials/beginner/basics/intro.html |
| 《动手学深度学习》英文推荐系统章 | MF、MovieLens、BPR 排序 | https://www.d2l.ai/chapter_recommender-systems/index.html |
| CS224W | 图表示/GNN 补充 | https://web.stanford.edu/class/cs224w/ |
| RecBole 官方文档 | 基线训练与评估 | https://recbole.io/docs/ |
| RecBole 通用推荐入门 | BPR 快速运行 | https://recbole.io/docs/get_started/started/general.html |
| RecBole 序列推荐入门 | SASRec 数据/协议参考 | https://recbole.io/docs/get_started/started/sequential.html |
| RecBole LightGCN | 图推荐模型配置 | https://recbole.io/docs/user_guide/model/general/lightgcn.html |
| RecBole SGL | 自监督模型配置 | https://recbole.io/docs/user_guide/model/general/sgl.html |
| RecBole SASRec | 序列模型配置 | https://recbole.io/docs/user_guide/model/sequential/sasrec.html |
| DirectAU 作者代码 | Loss 实现与 RecBole 集成参考 | https://github.com/THUwangcy/DirectAU |

> **链接说明**：此表以截至 2026-10-13 核对过的官方文档为主。此前提及的 `d2l.com.cn` 推荐系统中文章节未能稳定访问，因此改用官方英文第 21 章；PyTorch 基础仍优先使用中文教材。不同版本《动手学深度学习》的章节编号可能不同，按章节标题查找即可。

---

## 13. 阶段复盘模板（Week 1 在 Day 3；后续各阶段最后一天填写）

```markdown
### Week N 复盘
- 本周完成的任务（Day N–N）：
- 本周实际投入（小时）：
- 最难的 3 个知识点：
- 我能不看资料解释的核心公式：
- 我能不看答案独立写出的代码：
- 本周实验结果与可靠性问题：
- 尚未解决的 BUG / 疑问：
- 下周必须优先补的内容：
- 本周是否通过周验收：[ ] 是 [ ] 否
```

**最终原则：先把 BPR-MF 与 LightGCN 真正掌握，再扩展到 SGL、DirectAU 与 SASRec。阅读论文、写代码、训练评估和解释结果四者缺一不可。**
