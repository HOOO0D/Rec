"""第 7 天修订版：
两种采样方式。
LearnBPR 在 D_S = {(u,i,j): i in I_u+, j not in I_u+} 上均匀、有放回地采样。
每个正样本对采一个负样本不能替代此方法：用户的候选负样本数量不同时，两者赋予 (u,i) 的权重不同。
自动测试会检查这一区别。"""
from __future__ import annotations
import random


def build_positive_sets(train_pairs: list[tuple[int,int]], n_users: int) -> dict[int,set[int]]:
    """仅使用训练交互对构造映射，包含全部用户。"""
    # TODO A: 使用集合，无正样本用户对应空集合。
    dic = {u:set() for u in range(n_users)}
    for u,i in train_pairs:
        dic[u].add(i)
    return dic
    raise NotImplementedError


class BootstrapTripletSampler:
    def __init__(self, positive_by_user: dict[int,set[int]], n_items: int):
        """准备在 D_S 上均匀采样。公开整数 self.num_legal_triplets。
        无正样本或无合法负样本的用户贡献零个三元组。
        禁止构建 MovieLens 的全部三元组；本入门数据集允许 O(U*I) 候选负样本列表。
        用户 u 的权重为 |I_u+|*(I-|I_u+|)；
        按权重选择用户，再独立、均匀地选择该用户的正负样本。"""
        # TODO B: 构建可索引的用户、候选物品列表及累计权重。
        self.num_legal_triplets = 0
        self.users = []
        self.pos_candidates = []
        self.neg_candidates = []
        self.weight = []
        for user,pos in positive_by_user.items():
            self.users.append(user)
            self.pos_candidates.append(sorted(pos))
            self.neg_candidates.append([])
            idx = self.users.index(user)
            for item in range(n_items):
                if item not in pos:
                    self.neg_candidates[idx].append(item)
            w = len(pos)*len(self.neg_candidates[idx])
            self.weight.append(w)
            self.num_legal_triplets += w
        return 
        raise NotImplementedError

    def sample(self, num_samples: int, rng: random.Random) -> list[tuple[int,int,int]]:
        """有放回地独立抽取 num_samples 个三元组。
        结果有序，相同 rng 种子产生确定结果，允许重复三元组。
        空集合请求正数个样本或 num_samples 为负时抛出 ValueError；sample(0,rng) 返回 []。"""
        # TODO C: 三元组应在 D_S 上均匀分布。
        if self.num_legal_triplets == 0:
            raise ValueError("没有可采样的合法三元组")
        if num_samples<0:
            raise ValueError("采样数量不能为负数")
        if num_samples==0:
            return []
        ans = []
        for num in range(num_samples):
            # 带权重抽取
            u = rng.choices(self.users,self.weight)[0]
            idx = self.users.index(u)
            # 均匀抽取
            i = rng.choice(self.pos_candidates[idx])
            j = rng.choice(self.neg_candidates[idx])
            ans.append((u,i,j))
        return ans
        raise NotImplementedError
