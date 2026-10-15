"""Day2 学生实现区。只修改 TODO；验收运行 python Test_day2.py。
使用 CPU、torch.long ID；不需要下载数据。
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
def make_embedding(n, d):
    # TODO A1: 返回 n 行、d 列的 nn.Embedding，使用默认参数。
    embedding = torch.nn.Embedding(n,d)
    return embedding
    raise NotImplementedError("TODO A1")


def lookup(table, ids):
    # TODO A2: 使用 Embedding 查表，保持输入 ID 的维度。
    return table(ids)
    raise NotImplementedError("TODO A2")


class RatingDataset(Dataset):
    def __init__(self, rows):
        # TODO B1: 保存 rows，每行是 (user_id, item_id, rating)。
        self.rows = rows
        return
        raise NotImplementedError("TODO B1 __init__")

    def __len__(self):
        # TODO B1: 返回样本数。
        return len(self.rows)
        raise NotImplementedError("TODO B1 __len__")

    def __getitem__(self, index):
        # TODO B1: 返回三个标量 Tensor：long 用户ID、long 物品ID、float32 评分。
        user_ID, item_ID, rating = self.rows[index]
        return (
            torch.tensor(data=user_ID,dtype=torch.long),
            torch.tensor(data=item_ID,dtype=torch.long),
            torch.tensor(data=rating,dtype=torch.float32)
        )
        raise NotImplementedError("TODO B1 __getitem__")


def make_loader(dataset, batch_size, shuffle=False):
    # TODO C1: 返回 DataLoader；不丢尾批，num_workers=0。
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=0)
    raise NotImplementedError("TODO C1")


def pair_scores(user_table, item_table, user_ids, item_ids):
    # TODO D1: 两表分别查表，逐对计算点积：[B,d] 与 [B,d] -> [B]。
    # 注意 B=1 时仍返回 [1]；不要生成所有用户与物品的组合矩阵。
    scores = []
    # 逐对取出要用zip
    for id_u,id_i in zip(user_ids,item_ids):
       
       score =  (user_table(id_u) * item_table(id_i)).sum()
       scores.append(score)
    return torch.stack(scores)
    raise NotImplementedError("TODO D1")


def update_once(table, ids, optimizer):
    # TODO E1: 清梯度；查表并以所有输出元素之和为 loss；反传；step。
    Loss = table(ids).sum()
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()
    # 返回 Python float loss；不要在 step 后清梯度，验收需要检查 grad。
    return Loss.item()
    raise NotImplementedError("TODO E1")

