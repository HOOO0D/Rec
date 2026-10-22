"""第 7 天修订版：LearnBPR SGD 基线与 Adam 对照练习模板。主实验采用自助采样与 SGD；批量大于 1 时为小批次近似，batch_size=1 时每个三元组更新一次。
推荐命令：
  python train.py --dataset synthetic --optimizer sgd --lr 0.20 --epochs 50 --steps-per-epoch 216 --batch-size 8 --output-dir runs/synthetic_sgd
  python train.py --dataset synthetic --optimizer adam --lr 0.02 --epochs 50 --steps-per-epoch 216 --batch-size 8 --output-dir runs/synthetic_adam
  python train.py --dataset ml100k --data-path data/ml-100k/u.data --optimizer sgd --lr 0.05 --epochs 1 --batch-size 512 --output-dir runs/ml100k_sgd
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from bpr_model import BPRMF, bpr_loss
from negative_sampling import build_positive_sets, BootstrapTripletSampler


def synthetic_pairs() -> tuple[list[tuple[int,int]], int, int]:
    n_users, n_items = 6, 15
    return [(u, (2*u + step) % n_items) for u in range(n_users) for step in range(3)], n_users, n_items


def load_ml100k_ratings(data_path: str | Path) -> list[tuple[int,int,int,int]]:
    """读取制表符分隔的 MovieLens-100K u.data，将用户和物品 ID 转为从 0 开始。"""
    # TODO A: 返回四元组 (u, i, rating, timestamp)。
    records = []
    with Path(data_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            u,i,rating,timestamp = map(int,line.split())
            records.append((u-1,i-1,rating,timestamp))
    return records        
    raise NotImplementedError


def split_ratings(ratings: list[tuple[int,int,int,int]], valid_ratio: float,
                  test_ratio: float, seed: int
                  ) -> tuple[list[tuple[int,int]], list[tuple[int,int]], list[tuple[int,int]]]:
    """按用户分层，确定性地划分训练、验证和测试集。
    每个已观察 (u,i) 仅出现于一个子集；
    有记录的用户至少保留一个训练交互对。
    有至少 3 个不同物品时，每位用户的验证和测试集均应非空。
    使用 random.Random(seed) 打乱，不使用全局随机状态。
    返回排序后的交互对，ID 从 0 开始。"""
    # TODO B
# 建议在用户循环之前检查一次
    if not (0 <= valid_ratio < 1 and 0 <= test_ratio < 1):
        raise ValueError("验证和测试比例必须位于 [0, 1)")
    if valid_ratio + test_ratio >= 1:
        raise ValueError("验证和测试比例之和必须小于 1")
    rng = random.Random(seed)
    test_set = []
    train_set = []
    valid_set = []
    item_by_user = {}
    for u,i,_,_ in ratings:
        if u not in item_by_user:
            item_by_user[u]=set()
        item_by_user[u].add(i)

    for u in sorted(item_by_user):
        items = sorted(item_by_user[u])
        rng.shuffle(items)
        n = len(item_by_user[u])
        if n < 3:
            n_valid = 0
            n_test = 0
        else:
            n_valid=int(n*valid_ratio)
            n_test = int (n*test_ratio)
            if valid_ratio>0:
                n_valid = max(1,n_valid)
            if test_ratio:
                n_test = max(1,n_test)
            while(n_valid+n_test)>n-1:
                if n_valid>=n_test:
                    n_valid -= 1
                else:
                    n_test -= 1
        n_train = n-n_valid-n_test
        trains = items[:n_train]
        valids = items[n_train:n_train+n_valid]
        tests = items[n_train+n_valid:]
        valid_set.extend((u,i)for i in valids)
        train_set.extend((u,i)for i in trains)
        test_set.extend((u,i) for i in tests)

    return (sorted(train_set),sorted(valid_set),sorted(test_set))
    raise NotImplementedError


def train_one_epoch(model: BPRMF, sampler: BootstrapTripletSampler,
                    optimizer: torch.optim.Optimizer, batch_size: int,
                    steps_per_epoch: int, rng: random.Random,
                    l2_weight: float = 1e-4, device: str = 'cpu') -> dict:
    """有放回地采样 steps_per_epoch 个三元组再训练。
    每轮必须且只能调用一次 sampler.sample(steps_per_epoch, rng)，不得改为每个正样本采一个负样本。
    返回可序列化为 JSON 的字典：loss、bpr_loss、pairwise_acc、num_triplets、num_unique_triplets、sample_sha256（打乱前有序三元组的 SHA256 摘要）。
    验证和测试交互不得影响采样器。
    正则项为 l2_weight * mean(||user[u]||² + ||item[i]||² + ||item[j]||²)，仅对每批采样到的行计算，不对完整嵌入表计算。
    指标按实际批量大小加权平均。
    必须使用传入优化器，禁止内部创建 Adam。
    接近论文的设置为 torch.optim.SGD 与 batch_size=1。"""
    # TODO C: 生成三元组并计算可复现的 SHA256 摘要。
    triples = sampler.sample(num_samples=steps_per_epoch,rng=rng)
    sample_bytes = json.dumps(triples,separators=(",",":")).encode("utf-8")
    sample_sha256 = hashlib.sha256(sample_bytes).hexdigest()
    # TODO D: 批次转为张量；前向传播、计算损失和正则项、调用 zero_grad，
    #          再调用 backward、step；报告有限数值的指标。
    model.train()
    triples_tensor = torch.tensor(triples,dtype=torch.long,device=device)
    total_loss = 0.0
    total_bpr_loss = 0.0
    total_correct = 0.0
    total_count = 0
    for start in range(0,len(triples),batch_size):
        optimizer.zero_grad()
        batch = triples_tensor[start:start+batch_size]
        users = batch[:,0]
        pos = batch[:,1]
        neg = batch[:,2]
        u = model.user_embedding(users)
        p = model.item_embedding(pos)
        n = model.item_embedding(neg)
        (pos_score,neg_score) = model(users,pos,neg)
        loss = bpr_loss(pos_score,neg_score)
        regular = l2_weight *((u**2).sum(dim=1)
                              +(n**2).sum(dim=1)+(p**2).sum(dim=1)
                              ).mean()
        BPRLoss = loss + regular
        BPRLoss.backward()
        optimizer.step()
        B = batch.shape[0]
        total_loss += BPRLoss.item() * B
        total_bpr_loss += loss.item() * B
        total_correct += (pos_score>neg_score).sum().item()
        total_count += B
    return {
    "loss": total_loss / total_count,
    "bpr_loss": total_bpr_loss / total_count,
    "pairwise_acc": total_correct / total_count,
    "num_triplets": total_count,
    "num_unique_triplets": len(set(triples)),
    "sample_sha256": sample_sha256,
}
    raise NotImplementedError


def run_experiment(dataset: str, data_path: str | Path | None,
                   output_dir: str | Path, epochs: int = 1,
                   seed: int = 20261020, embedding_dim: int = 16,
                   batch_size: int = 512, lr: float = 0.05,
                   l2_weight: float = 1e-4, valid_ratio: float = 0.1,
                   test_ratio: float = 0.1, device: str = 'cpu',
                   optimizer_name: str = 'sgd', steps_per_epoch: int | None = None) -> dict:
    """执行 SGD 或 Adam 实验并保存复现记录。config.json 包含 seed、model、optimizer、lr、batch_size、steps_per_epoch、n_users/items、n_train/valid/test、sampler='bootstrap_uniform_with_replacement'、initial_weight_sha256、implicit_positive_policy。train_pairs.tsv、valid_pairs.tsv、test_pairs.tsv 使用从 0 开始的 ID。history.jsonl 每轮一条记录，包含 sample_sha256。checkpoint.pt 包含 state_dict、optimizer_state_dict、config。先在 torch.manual_seed(seed) 下初始化模型，再创建优化器。相同种子的 Adam 和 SGD 必须具有相同初始参数及三元组序列，学习率可不同。steps_per_epoch 默认为训练正样本对数量。所有已观察评分均视作隐式正反馈。测试集留给第 8 天，禁止用于第 7 天调参。"""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs 和 batch_size 必须大于 0")
    if dataset not in ("ml100k", "synthetic"):
        raise ValueError("不支持的数据集")
    if optimizer_name not in ("sgd", "adam"):
        raise ValueError("不支持的优化器")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    records = []
    n_users = 0
    n_items = 0
    if dataset == "ml100k":
        records = load_ml100k_ratings(data_path)
        n_users = max(u for u,i,r,t in records)+1
        n_items = max(i for u,i,r,t in records)+1
    else:
        pairs,n_users,n_items = synthetic_pairs()
        records = [(u,i,1,0) for u,i in pairs]

    train_pairs, valid_pairs, test_pairs = split_ratings(records,valid_ratio,test_ratio,seed)
    
    positive_user_items = build_positive_sets(train_pairs,n_users)
    sampler = BootstrapTripletSampler(positive_user_items,n_items)
    torch.manual_seed(seed)
    model = BPRMF(n_users,n_items,embedding_dim).to(device)
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    rng = random.Random(seed)
    if steps_per_epoch is None:
        steps_per_epoch = len(train_pairs)
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch 必须大于 0")

    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(
            tensor.detach().cpu().contiguous().numpy().tobytes()
        )
    initial_weight_sha256 = hasher.hexdigest()
    # 保存实验配置和固定的数据划分。
    config = {
        "dataset": dataset,
        "data_path": str(data_path) if data_path is not None else None,
        "seed": seed,
        "model": "BPRMF",
        "optimizer": optimizer_name,
        "lr": lr,
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "l2_weight": l2_weight,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "device": str(device),
        "n_users": n_users,
        "n_items": n_items,
        "n_train": len(train_pairs),
        "n_valid": len(valid_pairs),
        "n_test": len(test_pairs),
        "sampler": "bootstrap_uniform_with_replacement",
        "initial_weight_sha256": initial_weight_sha256,
        "implicit_positive_policy": "all_observed_ratings",
    }
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    for filename, pairs in [
        ("train_pairs.tsv", train_pairs),
        ("valid_pairs.tsv", valid_pairs),
        ("test_pairs.tsv", test_pairs),
    ]:
        with (out / filename).open("w", encoding="utf-8") as f:
            for u, i in pairs:
                f.write(f"{u}\t{i}\n")

    # 每轮写一条日志；同一个随机数生成器持续用于各轮采样。
    with (out / "history.jsonl").open("w", encoding="utf-8") as f:
        for epoch in range(epochs):
            stats = train_one_epoch(model,sampler,optimizer,batch_size,steps_per_epoch,rng,l2_weight,device)
            stats["epoch"] = epoch + 1
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")
            f.flush()

    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, out / "checkpoint.pt")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Day7 LearnBPR SGD vs Adam')
    parser.add_argument('--dataset', choices=['synthetic', 'ml100k'], required=True)
    parser.add_argument('--data-path', default='data/ml-100k/u.data')
    parser.add_argument('--output-dir', default='runs/day7')
    parser.add_argument('--optimizer', choices=['sgd','adam'], default='sgd')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--embedding-dim', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--l2-weight', type=float, default=1e-4)
    args = parser.parse_args()
    result = run_experiment(
        args.dataset, args.data_path, args.output_dir, epochs=args.epochs,
        seed=args.seed, embedding_dim=args.embedding_dim,
        batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch,
        lr=args.lr, l2_weight=args.l2_weight,
        optimizer_name=args.optimizer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
