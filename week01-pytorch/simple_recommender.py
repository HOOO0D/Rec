"""Day 3 练习模板：自行实现，不包含标准答案。

接口必须与 Test_Day3.py 一致。可以复用 Day 2 的知识，不要照抄模型实现。
"""
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


class SimpleRecommender(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        super().__init__()
        # TODO 1: 定义 self.user_embedding 和 self.item_embedding（均使用 nn.Embedding）
        # 今天只用两个 Embedding 做纯点积评分，不加 Bias/MLP。
        self.user_embedding = nn.Embedding(num_embeddings=n_users,embedding_dim=embedding_dim,dtype=torch.float32)
        self.item_embedding = nn.Embedding(num_embeddings=n_items,embedding_dim=embedding_dim,dtype=torch.float32)
        return
        raise NotImplementedError("TODO 1: 定义两个可训练的 Embedding 表")

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # TODO 2: user_ids、item_ids: [B]，返回逐样本点积结果 [B]
        # 注意 Batch=1 的情况。
        """
        逐对计算点积
        scores = []
        for (userId,itemId) in zip(user_ids,item_ids):
            score = (self.user_embedding(userId) * self.item_embedding(itemId)).sum()
            scores.append(score)
        return torch.stack(scores)
        """
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        return (u*i).sum(dim=1)
        raise NotImplementedError("TODO 2: 实现逐样本点积 forward")


def train_model(
    model: SimpleRecommender,
    dataloader: DataLoader,
    epochs: int = 200,
    lr: float = 0.05,
) -> list[float]:
    """MSE 训练；返回长度为 epochs 的每轮平均 Loss（Python float）。
    dataloader 每个 Batch 为 (user_ids, item_ids, ratings)。
    不在此函数中重新创建或替换模型对象；直接更新传入的 model。
    """
    # TODO 3: 建立 MSELoss / optimizer；每轮遍历 dataloader
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(),lr=lr)
    # TODO 4: 完成 zero_grad -> forward -> loss -> backward -> step
    Loss = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_sample = 0
        for u,i,r in dataloader:
            optimizer.zero_grad()
            l = loss(model(u,i),r)
            l.backward()
            optimizer.step()
            batch_size = r.size(0)
            total_loss += l.item() * batch_size
            total_sample += batch_size
        Loss.append(total_loss/total_sample)    
    # TODO 5: 累计每个 epoch 的平均 Loss（可按样本数加权），返回列表
    return Loss
    raise NotImplementedError("TODO 3-5: 实现完整训练循环")



def save_model(model: SimpleRecommender, path: str | Path) -> None:
    # TODO 6: 推荐保存 model.state_dict() 而不是整个 Python 模型对象
    torch.save(model.state_dict(),path)
    return
    raise NotImplementedError("TODO 6: 保存模型权重")


def load_model(
    path: str | Path,
    n_users: int,
    n_items: int,
    embedding_dim: int,
) -> SimpleRecommender:
    # TODO 7: 创建相同结构的新模型、读取权重、调用 load_state_dict、返回模型
    # 提示：torch.load(..., map_location='cpu', weights_only=True)
    model = SimpleRecommender(n_items=n_items,n_users=n_users,embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(path,weights_only=False))
    model.eval()
    return model
    raise NotImplementedError("TODO 7: 恢复模型权重")


if __name__ == "__main__":
    torch.manual_seed(42)
    n_users, n_items, embedding_dim = 5, 8, 3
    user_ids = torch.arange(n_users).repeat_interleave(n_items)
    item_ids = torch.arange(n_items).repeat(n_users)
    true_users = torch.randn(n_users, embedding_dim)
    true_items = torch.randn(n_items, embedding_dim)
    ratings = (true_users[user_ids] * true_items[item_ids]).sum(dim=-1)
    dataset = torch.utils.data.TensorDataset(user_ids, item_ids, ratings)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = SimpleRecommender(n_users, n_items, embedding_dim)
    losses = train_model(model, dataloader)
    print(f"Loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    model.eval()
    sample_users, sample_items = user_ids[:8], item_ids[:8]
    with torch.no_grad():
        scores = model(sample_users, sample_items)
    print("Scores:", scores)
    checkpoint_path = Path(__file__).with_suffix(".pt")
    save_model(model, checkpoint_path)
    restored_model = load_model(checkpoint_path, n_users, n_items, embedding_dim)
    with torch.no_grad():
        restored_scores = restored_model(sample_users, sample_items)
    print("Reloaded scores:", restored_scores)
    print("Scores match:", torch.allclose(scores, restored_scores))
