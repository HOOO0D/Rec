# Day 8｜BPR-MF 的 Top-K 全排序评价

> 学习目标：完成 Recall@K、Hit@K、NDCG@K，验证候选集过滤、数据独立性，接入 Day 7 的 MovieLens-100K 模型。  
> 原 RoadMap 对应 Day 8（Week 2），预计总投入 4–5 小时。  
> **先手算，再运行自动测试。不要提前查看最终测试集上的推荐指标。**

## 一、理论与手算（建议 60–75 分钟）

### 1. 必须能解释

- Recall@K 分子和分母是什么？为什么不是除以 K？
- Hit@K 与 Recall@K 的主要区别是什么？
- NDCG@K 为什么需要位置折扣？为什么还需要 IDCG？
- 为什么指标需要按用户计算，再对存在验证正例的用户取平均？
- Full Ranking 与采样候选排序的区别是什么？
- 为什么验证集评价必须过滤训练已交互物品？
- 为什么训练 Loss 下降不保证验证 Recall@K 一定提升？

### 2. 手算例题（先不要只抄代码结果）

模型给用户 U0 排出的前四名：`[3, 1, 0, 2]`。  
该用户验证集正物品集合：`{1, 2}`。

请独立写出：

1. Recall@3：
2. Hit@3：
3. DCG@3：
4. IDCG@3：
5. NDCG@3：
6. Recall@4、NDCG@4：
7. 如果 U0 的训练已交互物品为 `{3}`，上述排名哪里有问题？应该先进行什么处理？

**本次统一使用二值相关性：**

\[
\mathrm{Recall@K}(u)=\frac{|\mathrm{TopK}(u)\cap G_u|}{|G_u|}
\]

\[
\mathrm{Hit@K}(u)=\mathbb 1\bigl[\mathrm{TopK}(u)\cap G_u\neq\varnothing\bigr]
\]

\[
\mathrm{DCG@K}(u)=\sum_{r=1}^{K}\frac{\mathbb 1[i_r\in G_u]}{\log_2(r+1)}
\]

\[
\mathrm{IDCG@K}(u)=\sum_{r=1}^{\min(K,|G_u|)}\frac{1}{\log_2(r+1)}
\]

\[
\mathrm{NDCG@K}(u)=\frac{\mathrm{DCG@K}(u)}{\mathrm{IDCG@K}(u)}
\]

相关集合为空时，**单用户函数**按练习约定返回 0；**多用户评价器**直接跳过没有验证正例的用户，不计入用户平均值分母。

## 二、代码任务（建议 120 分钟）

`evaluate.py` 中共 7 个函数：

- [ ] `recall_at_k`：分母为留出正例数量
- [ ] `hit_at_k`：命中一个即为 1
- [ ] `ndcg_at_k`：正确使用位置折扣与理想排名
- [ ] `validate_splits`：集合内部去重、三份集合无重合
- [ ] `mask_seen_scores`：复制分数后将训练已见物品设为负无穷
- [ ] `topk_items`：过滤无效候选、同分按小物品 ID 优先
- [ ] `evaluate_full_ranking`：全部物品参与排序；在验证用户上做宏平均

本日禁止通过 RecBole 内置 evaluator 直接代替独立实现；Day 9 再进行源码和协议对照。

## 三、MovieLens-100K 评估协议（建议 45 分钟）

填写从 Day 7 的 `config.json` 实际读到的值：

| 项目 | 记录 |
|---|---|
| Day 7 实验目录 | |
| Day 7 优化器（SGD / Adam） | |
| 随机种子 | |
| 用户数/物品数 | |
| 训练交互数 | |
| 验证交互数 | |
| 测试交互数 | |
| 正反馈转换约定 | |
| 评价方式 | Full Ranking（固定） |
| 验证时屏蔽 | 仅训练正交互 |
| 候选数量 | 每用户全部物品减去其训练交互 |
| 主要 K | 10 |
| 验证指标与实验日志 | |

- [ ] 三份划分内部无重复，且交互对两两不重叠。
- [ ] `validation_metrics.json` 只有验证指标，没有测试指标。
- [ ] 保存 `evaluation_config.json` 和 `validation_recommendations.tsv`。
- [ ] 从推荐列表检查至少 5 位用户的训练已交互物品过滤。
- [ ] 对 SGD 和 Adam 比较时，使用同一份划分、同一物品全集、相同的 K 与过滤规则。

**特别注意：** Day 8 不允许用测试集调节模型。以后真正进行测试评价时，应清楚规定是否屏蔽 train + valid，并与对照系统采用一致协议。

## 四、自动验收与复盘（建议 30 分钟）

- 命令：`python Test_Day8.py`
- 得分（本地玩具数据）：____ / 80
- 命令：`python Test_Day8.py --run-dir runs/day7_sgd_ml100k`
- 得分（含 MovieLens Checkpoint）：____ / 100
- Recall@10：
- Hit@10：
- NDCG@10：
- 为什么不同指标可能走势不一致：
- 是否发现数据泄漏风险：
- 最难的一个实现细节：
- 第 9 天对照 RecBole 时首先要核对哪些协议：
