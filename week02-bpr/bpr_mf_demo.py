"""A minimal, runnable BPR-MF example (implicit feedback, leave-one-out evaluation)."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 2026
random.seed(SEED)
torch.manual_seed(SEED)

# Example: items clicked/liked by each user. No explicit ratings required.
interactions = {
    0: [0, 1, 2, 3, 4],
    1: [0, 1, 2, 3, 5],
    2: [1, 2, 3, 4, 5],
    3: [0, 2, 3, 4, 5],
    4: [8, 9, 10, 11, 12],
    5: [8, 9, 10, 11, 13],
    6: [9, 10, 11, 12, 13],
    7: [8, 10, 11, 12, 13],
}
num_users, num_items = 8, 16

# Hold out one observed positive per user for validation.
# Only training positives participate in the negative sampler.
rng = random.Random(SEED)
train, valid = {}, {}
for u, items in interactions.items():
    holdout = rng.choice(items)
    valid[u] = holdout
    train[u] = set(items) - {holdout}


class BPRDataset(Dataset):
    def __init__(self, train_items, n_items, seed=SEED):
        self.positive_pairs = [
            (u, i) for u, items in sorted(train_items.items()) for i in sorted(items)
        ]
        self.train_items = train_items
        self.n_items = n_items
        self.rng = random.Random(seed)
        assert all(len(items) < n_items for items in train_items.values())

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, index):
        u, i = self.positive_pairs[index]
        # Re-sample on each access; negative = not in this user's TRAIN positives.
        j = self.rng.randrange(self.n_items)
        while j in self.train_items[u]:
            j = self.rng.randrange(self.n_items)
        return u, i, j


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim=16):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.1)

    def forward(self, users, pos_items, neg_items):
        w = self.user_emb(users)
        h_pos = self.item_emb(pos_items)
        h_neg = self.item_emb(neg_items)
        delta = (w * (h_pos - h_neg)).sum(dim=-1)
        return delta, w, h_pos, h_neg

    def score_all(self, users):
        return self.user_emb(users) @ self.item_emb.weight.T


@torch.no_grad()
def evaluate(model, train_items, valid_items, k=3):
    model.eval()
    # For each user: rank held-out positive against all items not in TRAIN.
    scores = model.score_all(torch.arange(num_users))
    auc_total, hit_total = 0.0, 0.0
    for u, true_item in valid_items.items():
        scores_u = scores[u].clone()
        scores_u[list(train_items[u])] = float('-inf')
        topk = torch.topk(scores_u, k=k).indices.tolist()
        hit_total += int(true_item in topk)

        neg_indices = [
            j for j in range(num_items) if j not in train_items[u] and j != true_item
        ]
        neg_scores = scores_u[neg_indices]
        true_score = scores_u[true_item]
        auc_u = ((true_score > neg_scores).float().sum()
                 + 0.5 * (true_score == neg_scores).float().sum()) / len(neg_indices)
        auc_total += auc_u.item()
    n = len(valid_items)
    return hit_total / n, auc_total / n


def main():
    dataset = BPRDataset(train, num_items)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    model = BPRMF(num_users, num_items, dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    lambda_w = lambda_pos = lambda_neg = 1e-3

    hit, auc = evaluate(model, train, valid)
    print(f"Before training: HR@3={hit:.3f}, user-AUC={auc:.3f}")

    for epoch in range(1, 21):
        model.train()
        epoch_loss = 0.0
        for users, pos_items, neg_items in loader:
            delta, w, h_pos, h_neg = model(users, pos_items, neg_items)
            bpr_loss = -F.logsigmoid(delta).mean()
            # Regularize embeddings participating in this minibatch.
            reg_loss = 0.5 * (
                lambda_w * w.square().sum(dim=1)
                + lambda_pos * h_pos.square().sum(dim=1)
                + lambda_neg * h_neg.square().sum(dim=1)
            ).mean()
            loss = bpr_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch in (1, 5, 10, 20):
            hit, auc = evaluate(model, train, valid)
            print(f"Epoch {epoch:>3}: loss={epoch_loss/len(loader):.4f}, "
                  f"HR@3={hit:.3f}, user-AUC={auc:.3f}")

    # Produce recommendations for user 0, excluding training interactions.
    model.eval()
    with torch.no_grad():
        scores = model.score_all(torch.tensor([0])).squeeze(0)
        scores[list(train[0])] = float('-inf')
        print("User 0 top-5 candidates:", torch.topk(scores, k=5).indices.tolist())
        print("User 0 held-out positive:", valid[0])


if __name__ == "__main__":
    main()
