# Day 7 | BPR-MF 独立实现与验收

放到 `E:\AI\rec-learning\week02-bpr\`，保留 Day5 的 `model.py` 不覆盖。

本练习文件：
- `bpr_model.py`: 模型、评分、数值稳定 BPR Loss
- `negative_sampling.py`: 构建用户正样本表、负采样、三元组
- `train.py`: 解析 ml-100k、按用户留出、训练并保存 artifacts
- `Test_Day7.py`: 自动测试；A–D 本地合成数据，E 必须有真实 `u.data` 才能给满分
- `Day7_notes.md`: 理论与实验记录

## 下载 MovieLens-100K
访问 GroupLens 官方页面 https://grouplens.org/datasets/movielens/100k/ ；
下载 `ml-100k.zip`，解压，将其中的 `u.data` 放在：
`week02-bpr/data/ml-100k/u.data`。

> 不能替换为 ml-latest-small 的 `ratings.csv`。它不是同一批数据也不是同一格式。

## 命令
```powershell
conda activate E:\AI\envs\torch-learning
cd E:\AI\rec-learning\week02-bpr
python .\Test_Day7.py
python .\Test_Day7.py --ml100k .\data\ml-100k\u.data
python .\train.py --dataset synthetic --epochs 60 --output-dir .\runs\day7_synthetic
python .\train.py --dataset ml100k --data-path .\data\ml-100k\u.data --epochs 1 --output-dir .\runs\day7_ml100k
```

- 测试 A–D = 80 分，只有 E 的真正 ml-100k 训练及三份数据划分、日志落盘成功，才能拿满 100 分。
- 如果无真实数据，仅显示 80/100 + [SKIP] E；不算 Day7 全部完成。
- 不要求 Day 8 的 Recall@K/NDCG@K，不要求 Day 9 的 RecBole。
- 训练中不使用任何验证/测试交互作为监督，也不能拿验证/测试集的物品参与构建 `positives_by_user`；并非禁止这些验证物品作为未观察候选被抽样，这是现阶段的隐式反馈训练约定。
