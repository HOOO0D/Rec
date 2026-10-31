# 推荐系统科研入门：七阶段学习计划（LightGCN → Transformer → SASRec）

> **学习周期**：保留原版的阶段化节奏；Week 1 为 3 个学习日，其余每阶段 6 日，共 7 个阶段、39 个学习日（原版 33 日，新增 6 日）  
> **每日投入**：4–5 小时；建议以 4.5 小时安排  
> **当前起点**：截至 2026-10-30，已阅读 LightGCN 论文（原 Day 12），尚未完成 LightGCN 复现；Day 12 的完整验收仍以笔记和测试结果为准  
> **剩余任务**：若 Day 12 的阅读验收通过，从 Day 13 开始仍有 Day 13–39 共 27 个学习日，约 108–135 小时；未通过的验收另安排机动补练  
> **终点**：完成 BPR、LightGCN、Attention Is All You Need、SASRec、SGL、DirectAU 六篇论文的第一轮学习，具备独立实现核心算法、运行基线与分析实验的能力  
> **更新日期**：2026-10-30（v3：细化 Transformer 核心模块、输出层与 KV-Cache）

## 0. 先看这三条原则

1. **原理和代码并行**：每天至少 2 小时编程或可检验的公式/实验实践；不以“看完论文或视频”为验收标准。论文精读日可以用手算、源码审计和小型模块实验代替完整训练。
2. **不同论文的实践深度不同**：BPR-MF 和 LightGCN 尽可能独立实现；Attention Is All You Need 按 Token/Position Embedding → Attention → Add & Norm / FFN → Encoder/Decoder / 输出层 → KV-Cache 的依赖关系精读并实现核心模块，不要求复现翻译系统；SASRec 重点自主实现最小因果序列推荐模型并用 RecBole 运行、核对；SGL 在 LightGCN 上增量实现；DirectAU 重点实现 Loss。
3. **先正确，再涨指标**：评价协议、数据划分、防止数据泄漏、固定随机种子等优先于参数技巧。第一轮不要求复现六篇论文的全部表格，也不保证得到论文原始指标。

### 每日时间安排（以 270 分钟为例）

| 模块 | 时间 | 做什么 |
|---|---:|---|
| 理论学习 | 75 分钟 | 视频、教材、论文阅读、公式理解 |
| 编程实践 | 120 分钟 | 独立写代码、运行测试、排查错误；精读日可进行小型可验证实验 |
| 深入分析 | 45 分钟 | 对照源码、画结构图、阅读实验结果 |
| 整理复盘 | 30 分钟 | 记笔记、记录问题、提交 Git |

> Week 表示学习阶段，不严格对应自然周。Week 1 在 Day 3 后安排 1 天休息/机动；之后每完成 6 个学习日安排 1 天休息/机动。Day 编号不包含休息日。每日仍以 4–5 小时为上限，验收未通过则补练，不把欠账带到下一阶段。

### 本次重排规则与依赖门槛

- **Day 1–15 原任务、验收和编号保留**。目前名义进度是已阅读 Day 12；Day 13–15 仍需完成 LightGCN 的传播实现、BPR 训练、Full Ranking 评价和 0–4 层消融，不允许只读论文就视为完成。
- **新 Week 4（Day 16–21）**：精读 *Attention Is All You Need*，覆盖 Token Embedding、Position Encoding、Scaled Dot-Product / Multi-Head Attention、Add & Norm、FFN、Encoder/Decoder、输出投影；补充推理时的 KV-Cache，独立完成最小组件及前向整合。
- **新 Week 5（Day 22–27）**：紧接着学习 *Self-Attentive Sequential Recommendation*（SASRec），完成时序数据协议、最小模型训练、RecBole 基线及一次控制变量消融。
- **新 Week 6（Day 28–33）**：原 SGL 六天顺延，不压缩；**新 Week 7（Day 34–39）**：保留原 DirectAU 四天，并安排综合协议核对与六篇论文总结。
- **依赖门槛**：Day 15 通过后才进入 Transformer；Day 21 通过后才进入 SASRec；Day 27 通过后再进入 SGL。保留原 BPR/LightGCN 里程碑，并新增 Attention/SASRec 里程碑。
- **不同协议不做直接优劣比较**：MF、LightGCN、SGL、DirectAU 的 CF 随机划分和 SASRec 的时序下一物品预测必须各自固定协议；不能因为使用相同的 ml-100k 就直接比较 Recall@K/NDCG@K。

## 1. 七阶段总览（39 个学习日）

| 周次 | 主线 | 学习成果 / 交付物 | 关键验收 |
|---|---|---|---|
| 第 1 周（Day 1–3） | 已掌握基础 → Embedding → 训练整合 | 可训练的简易评分模型 | 不看教程独立写训练循环 |
| 第 2 周（Day 4–9） | 推荐系统基础 + BPR-MF | `week02-bpr/`、负采样、Top-K 评估 | 从 `(u,i,j)` 训练 Embedding 并评估 |
| 第 3 周（Day 10–15） | 图论/GNN + LightGCN | `week03-lightgcn/`、图传播及层数对比 | 独立实现、训练、评价 LightGCN |
| 第 4 周（Day 16–21） | Attention Is All You Need / Transformer | Token/Position、QKV、Multi-Head、Add & Norm、FFN、输出、Encoder/Decoder、KV-Cache | 手算与单测通过；能运行最小前向并验证增量缓存 |
| 第 5 周（Day 22–27） | SASRec + 时序推荐 | `week05-sasrec/`、数据协议、模型、基线、消融 | SASRec 可训练、可评价，时序协议正确 |
| 第 6 周（Day 28–33） | SGL + 对比学习 | 图增强、InfoNCE、联合训练、消融 | 实现 SGL-ED 简化版 |
| 第 7 周（Day 34–39） | DirectAU + 综合复盘 | Alignment / Uniformity Loss、对照和六篇论文总结 | 实现 DirectAU、协议审计与总验收 |

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
- [x] **Day 2｜Embedding 和批数据**：学习 `nn.Embedding`、`Dataset`、`DataLoader`、Batch 维度；从用户/物品 ID 得到向量并计算点积；完成 `embedding.py`。
  - **当日验收**：提交 embedding.py；一个批次的 ID 为 [B]、向量为 [B,d]、点积得分为 [B]；ID 使用整数类型；反传后被访问的 Embedding 行有梯度且更新，DataLoader 能遍历完整数据。
- [x] **Day 3｜周综合练习**：用两个 Embedding 表和一个简单 Loss 训练“用户—物品评分”小模型；检查参数确实发生更新；完成 `simple_recommender.py` 和 `week01_notes.md`。
  - **当日验收**：不照抄教程完成 simple_recommender.py；在固定种子的微型评分数据上观察 Loss 下降和参数更新；保存并重载后同一输入预测一致；提交 week01_notes.md 并通过本周全部验收。

### 周验收

- [x] 不看教程解释 `Tensor.shape`、广播、向量点积、矩阵乘法。
- [x] 能说出 `zero_grad()`、`backward()`、`step()` 的作用与顺序。
- [x] 能独立实现 `nn.Module` 的 `__init__` 和 `forward()`。
- [x] 能解释 Embedding 表为什么是可训练参数、用户 ID 为什么能映射成一个向量。
- [x] 能训练至少一个小模型，确认 Loss、梯度、参数更新过程合理。

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

- [x] **Day 4｜推荐基本概念**：复习召回、排序、UserCF、ItemCF、MF；明确显式/隐式反馈、已观察/未观察交互；写一页概念笔记。
  - **当日验收**：提交一页概念笔记，说明召回/排序、显式/隐式反馈、UserCF/ItemCF/MF；给出未交互不等于不喜欢的例子。
- [x] **Day 5｜矩阵分解**：理解用户/物品 Embedding、点积评分、正则化；使用 PyTorch 写 MF 评分函数；手动检查 `shape`。
  - **当日验收**：MF 支持批量用户/物品 ID；手算一组点积并与程序核对；说明正则化对象和评分输出形状。
- [x] **Day 6｜精读 BPR**：阅读论文第 1、3、4 节；理解训练三元组 `(u, i, j)`、BPR Loss、负采样；在纸上计算样例 Loss。
  - **当日验收**：写清 (u,i,j) 的约束；手算得分差为 0 时 Loss 为 log(2)，并验证正物品得分提高时 Loss 降低；记录论文假设。
- [x] **Day 7｜独立实现 BPR-MF**：编写模型、负采样器、训练循环；先在合成数据上验证，再在 MovieLens-100K 上训练；保存日志。
  - **当日验收**：负采样不取用户训练正物品，并处理无可采负例的用户；合成数据能学到正分高于负分；ml-100k 跑通一轮并保存划分、种子、配置与日志。
- [x] **Day 8｜评价指标**：学习 Recall@K、Hit@K、NDCG@K；实现 Top-K 评估并过滤训练集中已交互物品；保持验证/测试集独立。
  - **当日验收**：用手工排名验证 Recall、Hit、NDCG；过滤训练已交互物品，检查三份划分无交互重叠；固定候选集和 K，输出验证指标。
- [x] **Day 9｜RecBole 对照**：用 RecBole 运行 `BPR + ml-100k`；阅读内置 BPR 的 `calculate_loss()`；比较代码结构并写第一份实验记录。
  - **当日验收**：RecBole BPR 跑通并保存配置；读懂 calculate_loss；逐项核对划分、候选集、正则化和采样差异，协议不同则不作指标优劣结论。

### 核心公式

$$
\hat y_{ui}=\mathbf e_u^\top\mathbf e_i
$$

$$
\mathcal L_{\mathrm{BPR}}=-\log\sigma(\hat y_{ui}-\hat y_{uj})
$$

### 周验收

- [x] 不借助 RecBole，独立完成 BPR-MF 模型、训练及预测。
- [x] 知道为什么“未交互”不代表真实负反馈，理解负采样的局限。
- [x] 能对给定 Embedding 手算正负物品得分和 BPR Loss。
- [x] 能输出用户 Top-10 推荐，以及 Recall@K / NDCG@K 等指标。
- [x] 能解释训练 Loss、验证集指标和测试集指标的区别。

## 4. 第 3 周：LightGCN（Day 10–15）

**周目标**：学会用户—物品二部图构造、归一化邻居聚合和多层表示组合。

**主要资料**：

- [Stanford CS224W 课程主页](https://web.stanford.edu/class/cs224w/)：只选图表示、GNN、消息传递相关的公开讲义；不需要学完整门课。
- [RecBole LightGCN 文档](https://recbole.io/docs/user_guide/model/general/lightgcn.html)
- 上传的论文：**LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**

### 逐日清单

- [x] **Day 10｜图论与二部图**：理解节点、边、邻居、节点度、邻接矩阵；把用户—物品交互表转成二部图。
  - **当日验收**：用至少 2 个用户和 3 个物品画二部图；核对节点编号、双向边、邻接矩阵形状和节点度，图仅由训练交互构造。
- [x] **Day 11｜邻居聚合**：学习度矩阵 `D`、对称归一化邻接矩阵 `D^{-1/2} A D^{-1/2}`、消息传递、稀疏矩阵乘法。
  - **当日验收**：手算小图的一次对称归一化聚合，与稀疏乘法结果核对；对孤立节点不产生 NaN/Inf。
- [x] **Day 12｜精读 LightGCN**：学习论文第 3 节方法与消融实验；解释为何去掉特征变换和非线性激活，理解 0 层与多层表示。
  - **当日验收**：写出 LightGCN 与普通 GCN 的差异，以及 0 层、多层融合和 BPR 目标；结合论文消融说明结论适用范围。
- [ ] **Day 13｜图传播实现**：使用 PyTorch 实现 `E^(k+1) = A_norm @ E^(k)` 和不同层的 Embedding 融合；先用 2–3 个用户的小图手算验证。
  - **当日验收**：手算小图的 1–2 层传播并与代码比较；输出形状一致、梯度能回传到初始 Embedding；0 层退化为 MF 评分。
- [ ] **Day 14｜训练 LightGCN**：连接 BPR Loss、优化器和评估器，构建可运行的简化版 LightGCN。
  - **当日验收**：LightGCN 接入已验证的 BPR 训练和评估；小数据可过拟合、梯度有限、验证指标可输出；邻接矩阵未使用验证或测试边。
- [ ] **Day 15｜基线对比与消融**：运行 BPR-MF、LightGCN；固定数据划分和评估协议，比较 `n_layers = 0,1,2,3,4`；整理结果。
  - **当日验收**：固定协议比较 BPR-MF 与 LightGCN 的 0–4 层结果；记录种子、配置、验证选择与最终测试结果；能独立解释两者全流程才进入 Attention Is All You Need。

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

## 5. 第 4 周：Attention Is All You Need（Day 16–21）

**周目标**：按原论文 Figure 1 / Figure 2 的数据流，掌握 **Token Embedding → Positional Encoding → Scaled Dot-Product / Multi-Head Attention → Add & Norm → Position-wise FFN → Encoder / Decoder → Linear + Softmax 输出**；在理解自回归解码之后，补充 **KV-Cache** 的增量推理机制。能手算关键步骤、检查张量形状与掩码、独立实现简化组件，并组合出单层的 Encoder–Decoder 前向过程。**不要求复现 WMT 翻译结果或训练完整机器翻译系统。**

**阅读资料与准确性边界**：

- [原论文 *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)：优先精读 Abstract、Section 3.1–3.5、Figure 1–2；Section 1–2 和 Section 4–5 关注研究动机、复杂度与实验结论。
- [NeurIPS 原始论文 PDF](https://papers.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)：对照 Figure 1 的 Encoder / Decoder、Linear、Softmax、Shifted Right 和 Figure 2 的注意力结构。
- [Harvard Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)：学习 Embeddings、Positional Encoding、SublayerConnection、FFN、Encoder/Decoder、Generator 的 PyTorch 模块组成；这份资料是**后来的实现讲解**，以原论文架构为准。
- [《动手学深度学习》注意力机制](https://zh.d2l.ai/chapter_attention-mechanisms/index.html) 与 [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：补充 Softmax、QKV、缩放、Mask 与张量形状，对照自写实现。
- [Hugging Face — How caching works](https://huggingface.co/docs/transformers/en/cache_explanation)：**KV-Cache 是自回归推理工程扩展**，不是 2017 年原论文 Figure 1 中单独规定的训练模块；重点理解 Prefill、Decode、按层缓存 K/V、位置偏移、显存与计算代价。训练阶段默认不使用 KV-Cache。

**术语约定（必须正确区分）**：Token Embedding 是可学习的 token 向量；原论文 Position 使用正弦/余弦位置编码（并比较学习式位置表示），不能直接将二者混为同一个 Embedding 表。原始 Transformer 的 `Add & Norm` 是 `LayerNorm(x + Sublayer(x))`（Post-LN），不要误写成现代部分实现的 Pre-LN。原始 FFN 采用 `Linear → ReLU → Linear`，不是必须使用 GELU。输出层为 `Linear → Softmax` 得到下一 token 概率；训练用 `CrossEntropyLoss(logits, target)` 时直接传 logits，不要先重复 Softmax。

### 本周知识与交付物对照

| 必学内容 | 原论文位置 / 补充来源 | 核心产物与验收 |
|---|---|---|
| Token Embedding、Position | 3.4、3.5、Figure 1 | `embeddings_position.py`；正确解释 token/位置相加、位置变化、Padding、形状 |
| Scaled Dot-Product Attention / Mask | 3.2.1、3.2.3、Figure 2 | `scaled_attention.py`；手算分数/权重，Padding 与 Causal Mask 无泄漏 |
| Multi-Head Attention | 3.2.2 | `multihead_attention.py`；QKV 投影、拆头、拼头、W_O，检查形状和梯度 |
| Add & Norm、Position-wise FFN | 3.1、3.3 | `transformer_layers.py`；Post-LN、残差、逐位置 FFN 数值/梯度测试 |
| Encoder、Decoder、Cross-Attention、Output | 3.1、3.4、Figure 1 | `toy_transformer.py`；画架构图，跑通单层前向，输出 `[B,T,V]` logits |
| KV-Cache、增量解码 | Hugging Face 缓存教程（非原论文章节） | `kv_cache.py`；Prefill / Decode，缓存结果与不缓存因果推理一致 |

### 逐日清单

- [ ] **Day 16｜论文架构总览 + Token Embedding 与 Position**：阅读 Abstract、Section 1–2、3.4、3.5 和 Figure 1；明确词表 ID → Token Embedding（可训练）、`sqrt(d_model)` 缩放、正弦/余弦 Positional Encoding、Padding ID、Encoder 输入与右移后的 Decoder 输入，比较原论文位置编码与 SASRec 的可学习 Position Embedding。补充 Q/K/V 的来源和基本作用。编写 `embeddings_position.py`。
  - **当日验收**：输入 `[B,L]` 的 token ID 得到 `[B,L,d_model]`；不同位置编码可区分、相同位置编码在不同 Batch 一致；核对论文 `PE(pos,2i)` / `PE(pos,2i+1)` 数值；解释为何无 RNN/CNN 的模型必须注入顺序信息；对样例目标序列正确构造 Shifted-Right 输入，不能在预测当前 token 时把其本身提前输入。提交一页 Figure 1 数据流标注。
- [ ] **Day 17｜Scaled Dot-Product Attention + Padding/Causal Mask**：精读 Section 3.2.1、3.2.3；理解 Q/K/V、`QK^T/sqrt(d_k)`、Softmax、Value 加权和；区分 Encoder 双向 Self-Attention、Decoder 单向 Self-Attention、Cross-Attention 的查询与键值来源。独立实现 `scaled_attention.py`，支持 Padding Mask 与 Causal Mask。
  - **当日验收**：给定 2–3 个 token 手算得分矩阵、权重和输出并与代码对照；证明每个至少有一个有效 Key 的 Query 行权重之和为 1；被屏蔽位置权重为 0；在 `eval()` 下改变未来 token，前缀输出不变；对全被屏蔽 Query 行设置明确的稳定约定，不产生 NaN/Inf。输出形状、Mask 广播正确。
- [ ] **Day 18｜Multi-Head Attention 与线性投影**：精读 Section 3.2.2；学习 `W_Q/W_K/W_V/W_O`、`d_model=h*d_k`（此教学实现）、拆头、缩放、多头拼接和输出投影；明确 Self-Attention 中 Q/K/V 取自同一序列，Cross-Attention 中 Q 取自 Decoder、K/V 取自 Encoder。实现 `multihead_attention.py`（可复用 Day 17）。
  - **当日验收**：对 `[B,L,d_model]`、`h`、`d_k` 逐步列出 Q/K/V `[B,h,L,d_k]`、权重 `[B,h,L_q,L_k]`、拼头和输出 `[B,L_q,d_model]` 的形状；在 `d_model=8,h=2` 的玩具数据上前向与参考实现一致；`d_model % h != 0` 抛出清晰错误；W_Q/W_K/W_V/W_O 能获得有限梯度；能说明单头与多头的表达差别。
- [ ] **Day 19｜Add & Norm、Position-wise FFN 与 Encoder/Decoder 结构**：精读 Section 3.1、3.3 和 Figure 1；实现 `transformer_layers.py` 中的残差连接、原论文 **Post-LN** 形式 `LayerNorm(x + Dropout(Sublayer(x)))`、`Linear(d_model,d_ff) → ReLU → Linear(d_ff,d_model)`；画出 Encoder 每层 2 个子层、Decoder 每层 3 个子层（Masked Self-Attention → Cross-Attention → FFN），标注每个 Add & Norm；理解最终的 `Linear → Softmax`、权重共享与 `Outputs (shifted right)`。
  - **当日验收**：手算/代码核对小型残差和 LayerNorm；FFN 从 `[B,L,d_model]` 经 `[B,L,d_ff]` 回到 `[B,L,d_model]`；验证逐位置 FFN 不混合时间位置（在 `eval()` 下改变位置 j，不改变其他位置的 FFN 输出）；画图准确区分三种注意力的数据来源与 Mask；能解释 Decoder 为什么需要 Cross-Attention，为什么输出投影的 logits 维度是词表大小 V，为什么 `CrossEntropyLoss` 接受 logits。不得把 Post-LN 和 Pre-LN 视为相同实现。
- [ ] **Day 20｜KV-Cache 与自回归增量解码**：在 Day 17–18 的因果 Self-Attention 上，学习 Prefill 与逐 token Decode；为**每一层**保存 `[B,h,T_cache,d_k]` 的 K/V，不缓存历史 Q；新 token 只计算新 Q/K/V 并把 K/V 加入历史缓存。实现简化 `kv_cache.py`，先只针对 Decoder-only / 因果 Self-Attention；Encoder–Decoder 的 Cross-Attention K/V 缓存作为选做。不把 KV-Cache 误当成原论文明确规定的训练网络层。
  - **当日验收**：固定权重、`eval()`、关闭 Dropout，比较无缓存的逐前缀完整因果前向与缓存增量前向，最大绝对误差 `<=1e-5`（float32 玩具测试）；缓存长度逐步按已处理 token 数增长；显式测试位置编码偏移与 `Q_len=1,K_len=T_cache` 的 Mask 形状；解释 KV-Cache 减少重复计算、但增加显存占用，且不是训练梯度更新参数；记录至少一组无缓存/缓存重复计算量或运行时间对照（计时不作为硬性加速阈值）。
- [ ] **Day 21｜最小 Encoder–Decoder 整合 + 论文实验 + SASRec 衔接**：基于前五天的模块组合一个 `toy_transformer.py`：单层 Encoder Self-Attention + FFN、单层 Decoder Masked Self-Attention + Cross-Attention + FFN，以及输出投影 `Linear(d_model,V)`；精读原论文 Section 4–5，完成“问题—方法—公式—实验—局限”笔记。**只要求随机或玩具数据上的正确前向与反向，不训练 WMT 翻译系统。** 明确 SASRec 只复用 Item/Position、因果 Self-Attention、FFN 等机制，不直接使用完整机器翻译 Encoder–Decoder 或必须启用 KV-Cache。
  - **当日验收**：Encoder 输入 `[B,S]`、右移 Decoder 输入 `[B,T]`，输出 logits `[B,T,V]`，维度正确；Decoder Cross-Attention 权重 `[B,h,T,S]`，能识别 Q 与 K/V 来源；更改未来 Decoder token 不改变较早位置 logits（`eval()`、固定 Encoder 输入）；Padding 处理合理，Loss 有限、反传可更新参数；提交 `transformer_architecture.md`、`attention_paper_notes.md`，运行 `Test_Transformer.py`，能脱稿解释本周各模块与 SASRec 的联系及区别。

### 核心公式

**Embedding 与位置编码**（原论文使用正弦/余弦位置编码；与可学习 Token Embedding 相加）：

$$
X=\sqrt{d_{model}}\,\mathrm{Embedding}(tokens)+PE(positions)
$$

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

**缩放点积、Mask 与多头注意力**：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(head_1,\ldots,head_h)W^O
$$

**Add & Norm（原论文 Post-LN）与逐位置前向反馈网络**：

$$
\mathrm{AddNorm}(x)=\mathrm{LayerNorm}(x+\mathrm{Dropout}(\mathrm{Sublayer}(x)))
$$

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

**输出层与训练约定**：

$$
\mathrm{logits}=H_{decoder}W_{vocab}+b,\quad
p(y_t\mid y_{<t},src)=\mathrm{Softmax}(\mathrm{logits}_t)
$$

**KV-Cache（增量推理扩展，而非原论文独立训练层）**：

$$
K_{cache}\leftarrow [K_{past};K_{new}],\quad V_{cache}\leftarrow[V_{past};V_{new}]
$$

### 周验收与里程碑（100 分）

| 模块 | 验收内容 | 分值 |
|---|---|---:|
| A. Embedding / Position | Token Embedding、正弦位置编码、右移输入、形状与 Padding | 15 |
| B. Attention / Mask | 缩放点积、Softmax、Padding/Causal Mask、未来泄漏测试 | 20 |
| C. Multi-Head | 四组投影矩阵、拆头拼头、形状与梯度 | 15 |
| D. Block / Architecture / Output | Post-LN、FFN、Encoder/Decoder 三种注意力、输出 logits | 25 |
| E. KV-Cache | 每层 K/V 缓存、Prefill/Decode、与无缓存推理数值一致 | 15 |
| F. 论文与迁移 | 原论文结构/实验/局限笔记、与 SASRec 的联系及差异 | 10 |
| **合计** | **全部模块均需完成，建议 85/100 及以上** | **100** |

- [ ] 能从 Figure 1 不看原图画出 Encoder、Decoder、残差/Add & Norm、FFN、Cross-Attention 和 Linear + Softmax 输出路径。
- [ ] 能逐步写出 `[B,L,d_model] → Q/K/V → [B,h,L,d_k] → Attention → [B,L,d_model]` 的维度变化。
- [ ] 能解释原论文正弦位置编码与 SASRec 可学习位置 Embedding 的区别，并验证 Shifted-Right / Causal Mask 不泄漏未来 token。
- [ ] 能写出 Post-LN 与 Pre-LN 的不同、残差作用和 FFN 的逐位置性。
- [ ] KV-Cache 仅用于推理，能验证数值等价并解释为什么不能简单复用整个历史序列的 Q。
- [ ] 最小 Encoder–Decoder 前向、反向测试和论文理解通过，且能够解释哪些模块被 SASRec 复用、哪些不是其必需组件。

> **里程碑 B（Day 21）**：完成全部核心模块的理论说明与可检验实现，并拿到至少 85/100；因果 Mask、张量形状、位置偏移和 Encoder/Decoder 路径属于不可跳过项。KV-Cache 只作为增量推理基础，不延伸至 PagedAttention、FlashAttention 内核和显存分页。通过后进入 SASRec，不重复再安排完整 Transformer 基础阶段。

## 6. 第 5 周：SASRec 与时序推荐（Day 22–27）

**周目标**：在 *Attention Is All You Need* 的基础上，精读 *Self-Attentive Sequential Recommendation*，将因果 Self-Attention 用于用户行为序列的下一物品预测；完成最小可训练模型、RecBole 标准实现和一组消融。

**主要资料**：

- [原论文：Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)：重点任务动机、模型结构、训练目标、实验设置与消融结果。
- [RecBole SASRec 文档](https://recbole.io/docs/user_guide/model/sequential/sasrec.html) 与 [GitHub 文档镜像](https://github.com/RUCAIBox/RecBole/blob/master/docs/source/user_guide/model/sequential/sasrec.rst)
- [RecBole 序列推荐 Quick Start](https://recbole.io/docs/get_started/started/sequential.html)
- 第 4 周的 `embeddings_position.py`、`scaled_attention.py`、`multihead_attention.py`、`transformer_layers.py`、`toy_transformer.py`（复用 Embedding、Attention、Add & Norm、FFN 核心逻辑；标准离线 SASRec 训练通常无需使用 `kv_cache.py`，不直接照搬完整翻译 Transformer）。

### 逐日清单

- [ ] **Day 22｜序列推荐与时间协议**：学习行为顺序、下一物品预测、时间戳、序列截断和 Padding；用 MovieLens-100K 的时间戳构建有序历史，区分时序留出与原 CF 随机划分。
  - **当日验收**：构造 2–3 位用户的交互序列，按时间列出输入/下一物品目标；处理不足长度的序列、截断、Padding 和同时间戳稳定排序；训练输入不得包含留出目标或其后发生的行为。
- [ ] **Day 23｜精读 SASRec 论文**：理解从 Markov Chain / RNN 到 Self-Attention 的设计动机，逐项映射 Transformer 的 Token Embedding→Item Embedding、Position、因果 Self-Attention、Add & Norm、FFN、预测层；说明为什么不使用机器翻译 Decoder 的 Encoder–Decoder Cross-Attention。
  - **当日验收**：提交 `sasrec_paper_notes.md` 和带张量维度的模型结构图；说明 SASRec 与原 Transformer Encoder–Decoder 的区别；解释 Causal Mask 与位置 Embedding 的必要性，并记录原论文的训练和评估协议。
- [ ] **Day 24｜SASRec 核心模块**：基于第 4 周已测试的模块实现 `sasrec_core.py`，包括物品/位置 Embedding、因果 Attention、Padding 屏蔽、FFN、残差/LayerNorm 和序列末端有效位置表示提取；明确使用的 Norm 顺序以 SASRec 实现与源码为准。
  - **当日验收**：测试输入 `[B,L]` 到隐藏表示 `[B,L,d]`、最终用户状态 `[B,d]` 的形状；改变未来物品不影响前缀输出；Padding 位置不泄漏信息，空历史采取明确约定且无 NaN。
- [ ] **Day 25｜最小 SASRec 训练闭环**：用固定种子玩具序列构造下一物品训练样本，组合 SASRec Encoder 与候选物品评分、损失函数、反向传播和保存/加载。
  - **当日验收**：至少运行一个可学习的小数据集并观察 Loss 有效下降；参数和梯度有限；训练输入不包含目标物品；明确记下采用的是采样二元交叉熵、BPR 还是全物品 CE，不能混称为“论文完全一致”。
- [ ] **Day 26｜RecBole SASRec + ml-100k**：使用 RecBole 运行标准 SASRec，检查 `timestamp` 字段加载、`order: TO` 与时序留出协议、`MAX_ITEM_LIST_LENGTH`、Padding、因果掩码、损失及评价候选模式。
  - **当日验收**：模型训练和验证、测试流程跑通；保存配置、数据处理、划分、种子、模型 Checkpoint 和 Recall@K/NDCG@K；核对 `loss_type`（RecBole 可配置 CE 或 BPR）与原论文的训练目标差异。
- [ ] **Day 27｜SASRec 对照与消融**：保持时序协议不变，只改变序列最大长度或注意力层数；分析验证集变化，完成 Week 5 复盘并与 Transformer 论文作结构对照。
  - **当日验收**：至少完成基线 + 一组受控变体，保存结果、配置和原因分析；验证集负责选参数，测试集只用于最终报告；不把时序 SASRec 的指标与随机划分 CF 指标直接比较。

### 特别注意：SASRec 的实验协议

- SASRec 的对象是按时间排序的用户交互序列，必须建立输入历史与下一物品目标的时间边界。
- 原论文 SASRec 训练目标、RecBole SASRec `loss_type=CE` 或 `BPR` 和你手写的 BPR-MF 目标不能仅凭名称视为一致；必须以对应源码/配置实际设置为准。
- Padding 是模型输入的占位符，不能成为真正的推荐候选；因果掩码必须防止读取未来物品。
- SASRec 的下一物品预测协议与 CF 的随机交互划分不同；同一 MovieLens 数据集也不能直接横向比较指标。标准离线 SASRec 的序列编码通常按整段历史进行，并不以 KV-Cache 为必要前提；只有增量在线预测的特殊实现才需要另外讨论缓存、位置和掩码一致性。

### 核心公式

$$
X=\operatorname{ItemEmbedding}(s)+\operatorname{PositionEmbedding}(s)
$$

$$
\operatorname{CausalAttention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M_{\mathrm{causal}}\right)V
$$

$$
\hat y_{u,i\mid s}=\mathbf h_{u,s}^\top\mathbf e_i
$$

### 周验收

- [ ] 不看资料解释 SASRec 的输入序列、每个位置的预测目标、位置与因果掩码。
- [ ] 独立实现最小因果序列推荐模块，梯度正常，输入不读取未来目标。
- [ ] 用 RecBole 在 ml-100k 完成一次时序训练与评价，并保存可复现实验记录。
- [ ] 固定时序协议完成至少一次参数消融，并能解释原论文与框架实现的差异。

> **里程碑 C（Day 27）**：从原始时间戳到时序数据集、SASRec 核心实现、训练和评估全部打通后，才进入 SGL。不会为了赶进度跳过时间泄漏检查。

## 7. 第 6 周：SGL 与对比学习（Day 28–33）

**周目标**：在已经完成的 LightGCN 上加入图增强和 InfoNCE 对比损失，理解自监督辅助训练。此阶段沿用原版 Week 4 的六项任务与验收，**只顺延编号，不压缩内容**。

**主要资料**：

- [RecBole SGL 文档](https://recbole.io/docs/user_guide/model/general/sgl.html)
- 上传的论文：**Self-supervised Graph Learning for Recommendation**
- Day 13–15 的 LightGCN 图传播、训练器和 CF 评价模块。

### 逐日清单

- [ ] **Day 28｜自监督与对比学习**：理解自监督任务、正负样本对、两种数据视图、对比学习的监督信号。
  - **当日验收**：画出同一节点两视图的正对及其他节点的负对；解释监督信号来源，列出与 BPR 负物品的区别。
- [ ] **Day 29｜InfoNCE**：补充余弦相似度、Softmax、温度参数 `tau`、Hard Negative；手算一个小批次的对比损失。
  - **当日验收**：手算小批次余弦相似度和 InfoNCE，与程序核对；改变 tau 比较分布；使用稳定计算并确认损失和梯度有限。
- [ ] **Day 30｜精读 SGL**：学习图增强的 Node Dropout / Edge Dropout / Random Walk，以及主任务与自监督任务关系。
  - **当日验收**：整理三种图增强及主任务/辅助任务关系；明确本次实现 SGL-ED 的范围和与论文实现的差异。
- [ ] **Day 31｜图增强代码**：在 LightGCN 上实现 Edge Dropout，构造两个独立增强图视图；检查保留边比例。
  - **当日验收**：两视图独立采样、节点编号一致；无向边成对保留/删除；记录实际保留比例，drop_ratio=0 时恢复原图并重新计算归一化。
- [ ] **Day 32｜联合训练**：实现 InfoNCE Loss，联立 BPR Loss 和自监督 Loss，完成 SGL-ED 简化版。
  - **当日验收**：联合 Loss 可反传至 Embedding；分别记录 BPR/SSL/总 Loss；ssl_weight=0 时与相同配置 LightGCN 训练目标一致。
- [ ] **Day 33｜对照与消融**：用 RecBole SGL 对照；测试不同 `ssl_weight` 或 `drop_ratio`，注意逐次只控制一个变量。
  - **当日验收**：跑通 RecBole SGL 对照；一次仅改变 ssl_weight 或 drop_ratio，至少基线加一个变体；保存协议差异、结果表和原因分析。

### 核心公式

$$
\mathcal L_{\mathrm{SGL}}=\mathcal L_{\mathrm{BPR}}+\lambda_{\mathrm{ssl}}\mathcal L_{\mathrm{InfoNCE}}+\mathcal L_{\mathrm{reg}}
$$

### 周验收

- [ ] 能清楚区分 BPR 中的负物品与 InfoNCE 中的对比负样本。
- [ ] 能解释同一个节点两个增强表示为何构成正样本对。
- [ ] 能解释温度参数对分布和难负样本权重的影响。
- [ ] 简化版 SGL-ED 能训练、评估，并有至少一个消融结果。

## 8. 第 7 周：DirectAU 与综合总结（Day 34–39）

**周目标**：保留原版 DirectAU 的四个学习任务和验收；补充跨论文的协议整理与总结，形成可复查的推荐系统入门研究成果。

**主要资料**：

- [DirectAU 论文作者仓库](https://github.com/THUwangcy/DirectAU)
- 上传的论文：**Towards Representation Alignment and Uniformity in Collaborative Filtering**
- 原版 BPR、LightGCN、SGL 和新阶段 SASRec 的代码、日志与论文笔记。

### 逐日清单

- [ ] **Day 34｜表示空间基础**：学习 L2 归一化、向量距离、Alignment、Uniformity；解释为什么只追求正样本靠近会有表示坍缩风险。
  - **当日验收**：对一组向量手算 L2 归一化、距离；用全部向量相同的例子说明 Alignment 好但 Uniformity 差，记录坍缩风险。
- [ ] **Day 35｜精读 DirectAU**：理解论文理论动机与第 4 节优化目标；和 BPR、SGL 进行比较。
  - **当日验收**：用自己的话写清 DirectAU 目标、正交互取样和两项损失的作用；列出与 BPR、SGL 的区别及论文局限。
- [ ] **Day 36｜实现 DirectAU**：使用 PyTorch 编写 Alignment Loss、Uniformity Loss，在 MF Encoder 上训练。
  - **当日验收**：独立实现两项 Loss 并与手工小例核对；处理批次不足两个有效向量的情形，损失/梯度有限；MF Encoder 可完成训练。
- [ ] **Day 37｜对照实验**：对照论文作者源码，在同一数据划分和评估协议下比较 BPR 与 DirectAU；记录 `gamma` 影响。
  - **当日验收**：核对作者源码的归一化、距离及权重定义；固定 CF 协议比较 BPR 与 DirectAU，至少完成一组 gamma 对照并保存结果。
- [ ] **Day 38｜跨模型实验与协议审计**：统一整理 BPR-MF、LightGCN、SGL、DirectAU 的 CF 协议、模型参数和消融结果；另用独立表格记录 SASRec 的时序实验；提炼各模型适用场景与实验差异。
  - **当日验收**：输出 `protocol_audit.md` 和 `comparison.md`；同协议比较时逐项核对正例、划分、候选集和过滤规则；不同协议的结果分表报告、不跨协议宣称某模型最优。
- [ ] **Day 39｜六篇论文总结与最终验收**：补齐 BPR、LightGCN、Attention Is All You Need、SASRec、SGL、DirectAU 六篇论文的“问题—方法—公式—实验—局限”；整理代码仓库、模型配置、结果表及待解决问题。
  - **当日验收**：六篇论文各有对应笔记，代码、配置、日志、Checkpoint 和结果可追溯；逐项核对最终能力验收并列出未通过项和补练计划，完成 `final_report.md`。

### 核心公式

$$
\mathcal L_{\mathrm{DirectAU}}=\mathcal L_{\mathrm{align}}+\gamma\mathcal L_{\mathrm{uniform}}
$$

### 周验收

- [ ] 可以解释 Alignment、Uniformity 分别优化什么，以及表示坍缩的具体例子。
- [ ] 能自己实现并运行 DirectAU Loss，而不只是调用作者仓库。
- [ ] 能在一致的 CF 协议下比较 BPR 与 DirectAU，并记录至少一组 `gamma` 消融。
- [ ] 六篇论文均完成“问题—方法—公式—实验—局限”的结构化笔记。
- [ ] 完成 CF 与 SASRec 时序协议的独立对照，不将两类结果混为一谈。

## 9. 七阶段最终能力验收（11 项）

- [ ] **PyTorch**：不看教程定义 `nn.Module`，完成训练循环。
- [ ] **BPR-MF**：实现训练三元组、负采样、BPR Loss 和 Embedding 参数学习。
- [ ] **评价指标**：正确实现 Recall@K、NDCG@K，并解释不同评估协议的限制。
- [ ] **LightGCN**：能构造归一化邻接矩阵，实现多层图传播和 BPR 训练。
- [ ] **Transformer / Attention Is All You Need**：能实现 Token/Position、缩放与多头注意力、Add & Norm、FFN、Encoder/Decoder 简化前向、Linear 输出；理解原始 Post-LN，能验证 Causal Mask 与 KV-Cache 增量推理的一致性。
- [ ] **SASRec**：能正确准备时序下一物品数据、实现因果掩码、训练并评价序列推荐模型。
- [ ] **RecBole**：会通过配置运行 BPR、LightGCN、SGL 和 SASRec，理解主要接口及协议差异。
- [ ] **SGL**：能实现 Edge Dropout 和简化 InfoNCE，并进行联合训练。
- [ ] **DirectAU**：能实现 Alignment/Uniformity Loss 并解释表示坍缩。
- [ ] **科研实验**：至少做过一次公平的基线对比、一次控制变量消融，能够识别时序数据泄漏。
- [ ] **论文表达**：能脱离原文介绍六篇论文各自解决的问题、方法、创新和限制。

**优先级最高的门槛**：Day 15 时确保 BPR-MF / LightGCN 的“原理—实现—训练—评价”闭环；Day 21 时确保 Attention / Causal Mask、Encoder–Decoder 与 KV-Cache 机制正确；Day 27 时确保 SASRec 的时序训练与评价可靠。如果没有达到，宁可增加机动天数，也不要只为了赶进度进入下一阶段。

## 10. 预期项目目录

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
├── week04-transformer/
│   ├── embeddings_position.py
│   ├── scaled_attention.py
│   ├── multihead_attention.py
│   ├── transformer_layers.py
│   ├── kv_cache.py
│   ├── toy_transformer.py
│   ├── Test_Transformer.py
│   ├── transformer_architecture.md
│   └── attention_paper_notes.md
├── week05-sasrec/
│   ├── sequence_data.py
│   ├── sasrec_core.py
│   ├── train.py
│   ├── recbole_baseline/
│   └── experiments/
├── week06-sgl/
│   ├── augmentation.py
│   ├── contrastive_loss.py
│   └── experiments/
├── week07-directau/
│   ├── losses.py
│   ├── protocol_audit.md
│   └── experiments/
├── configs/
├── results/
└── README.md
```

> 新目录是**建议的重排后命名**；如果已经有 `week04-sgl/`、`week05-directau/` 或 `week06-sasrec/` 等旧目录，请保留原文件，不要覆盖。后续任务只需建立自己的映射关系并在实验记录中写清楚路径。训练配置、随机种子、指标和日志路径建议每阶段提交；大型权重和中间数据不直接放入 Git。

## 11. 每次实验必须记录的最小信息

以下模板可直接复制到阶段 `notes.md`：

```markdown
### 实验记录：YYYY-MM-DD
- 任务与假设：
- 模型 / 对照模型：
- 数据集及版本：
- 数据过滤与训练/验证/测试划分：
- 全排序 / 采样排序与候选集：
- 已交互物品过滤方式：
- 若为序列推荐：排序时间字段、时间戳同值处理、训练历史/目标边界、Padding 与 Causal Mask：
- Embedding 维度、层数、学习率、Batch Size、正则化：
- 负采样方式 / 序列训练目标（BCE、BPR、CE 等）：
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
- **严格区分 CF 随机交互划分和 SASRec 时序下一物品协议**。验证结果若不可比，应分表列出，不强行排序。
- **模型没涨指标不代表实现错误**；先排查实现和协议，再分析优化、模型归纳偏置、数据特性等原因。

## 12. 学习中需要暂时忽略的内容

| 内容 | 当前建议 |
|---|---|
| 冷启动 | 了解新用户、新物品冷启动的概念，不必现在专门复现论文 |
| 工业特征工程 / 线上涨指标技巧 | 暂缓，先掌握规范的离线实验和核心算法 |
| 大规模部署、分布式训练、多 GPU | 暂缓，入门数据集用 CPU 或单 GPU 即可 |
| 完整学习 CS224W | 没必要；只选择图表示与消息传递部分 |
| 完整复现 Transformer 机器翻译系统与大模型预训练 | 没必要；今天要求的是可验证的单层 Encoder–Decoder 前向与模块学习，而非 WMT 大规模训练 |
| KV-Cache 进阶内核与系统优化 | 第一轮只实现基础逐层 K/V 缓存及数值一致性；PagedAttention、FlashAttention 内核、多 GPU 推理暂缓 |
| 六篇论文所有证明、全部实验表格 | 第一轮不强求；保证核心公式、模型机制与重要结果理解即可 |

## 13. 官方参考资源汇总

| 资源 | 用途 | 链接 |
|---|---|---|
| 《动手学深度学习》中文 | PyTorch 与数学基础 | https://zh.d2l.ai/ |
| PyTorch Learn the Basics | Tensor、模型、Autograd、Optimizer | https://docs.pytorch.org/tutorials/beginner/basics/intro.html |
| 《动手学深度学习》英文推荐系统章 | MF、MovieLens、BPR 排序 | https://www.d2l.ai/chapter_recommender-systems/index.html |
| CS224W | 图表示/GNN 补充 | https://web.stanford.edu/class/cs224w/ |
| RecBole 官方文档 | 基线训练与评估 | https://recbole.io/docs/ |
| RecBole 通用推荐入门 | BPR 快速运行 | https://recbole.io/docs/get_started/started/general.html |
| RecBole LightGCN | 图推荐模型配置 | https://recbole.io/docs/user_guide/model/general/lightgcn.html |
| Attention Is All You Need 论文 | Transformer 结构、Self-Attention、Multi-Head 与 Mask | https://arxiv.org/abs/1706.03762 |
| Attention Is All You Need 官方 PDF | Figure 1 / Figure 2、方法与训练实验 | https://papers.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf |
| PyTorch Scaled Dot Product Attention | 注意力代码与形状对照 | https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html |
| Harvard Annotated Transformer | 逐组件对照 Embedding、Encoder、Decoder、FFN、输出层代码 | https://nlp.seas.harvard.edu/annotated-transformer/ |
| Hugging Face KV-Cache 教程 | 推理 Prefill/Decode、按层 K/V 缓存与掩码位置偏移 | https://huggingface.co/docs/transformers/en/cache_explanation |
| SASRec 论文 | 自注意力序列推荐原始方法 | https://arxiv.org/abs/1808.09781 |
| RecBole 序列推荐入门 | SASRec 数据/协议参考 | https://recbole.io/docs/get_started/started/sequential.html |
| RecBole SASRec | 图示、训练与评价配置 | https://recbole.io/docs/user_guide/model/sequential/sasrec.html |
| RecBole SASRec GitHub 文档 | 官方 Docs 无法访问时的替代 | https://github.com/RUCAIBox/RecBole/blob/master/docs/source/user_guide/model/sequential/sasrec.rst |
| RecBole SGL | SGL 自监督图推荐配置 | https://recbole.io/docs/user_guide/model/general/sgl.html |
| DirectAU 作者代码 | Loss 实现与 RecBole 集成参考 | https://github.com/THUwangcy/DirectAU |

> **链接说明**：原版 2026-10-13 核对过的资源保留，新增 Attention Is All You Need、SASRec 论文和 PyTorch 注意力官方文档。由于 RecBole Docs 可能访问不稳定，优先同时保留 GitHub 文档镜像。不同版本教材章节编号可能不同，按章节标题查找即可。

---

## 14. 阶段复盘模板（Week 1 在 Day 3；其余每阶段最后一天填写）

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

**最终原则：先把 BPR-MF 与 LightGCN 真正掌握，再精读 Transformer 并实现 SASRec，之后扩展到 SGL 和 DirectAU。阅读论文、写代码、训练评估和解释结果四者缺一不可。**
