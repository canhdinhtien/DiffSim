"""Discriminative score (Yoon et al., NeurIPS 2019), PyTorch port of the TimeGAN metric.

A post-hoc GRU classifier separates real vs synthetic sequences; score = |0.5 - test accuracy| (lower is better).
Protocol as in the TimeGAN code: 80/20 split of each set, Adam, `iterations` mini-batches of `batch_size`,
classify from the last hidden state. Deviation (shared by all 8 methods): 2-layer GRU with hidden_dim 16 instead of
1 layer with hidden_dim = dim/2 (= 3 for S1, which is too weak a classifier and gives optimistic scores).
Inputs: equal-length arrays (N, L, F), already min-max normalized.
"""
import numpy as np
import torch
from torch import nn


def batch_generator(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb


class _GRUHead(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, out_dim, per_step):
        super().__init__()
        self.rnn = nn.GRU(dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.per_step = per_step

    def forward(self, x):
        out, h = self.rnn(x)
        return self.fc(out if self.per_step else h[-1])


def discriminative_score_metrics(ori_data, generated_data, iterations=2000, batch_size=128, hidden_dim=16,
                                 num_layers=2, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ori, gen = np.asarray(ori_data, np.float32), np.asarray(generated_data, np.float32)

    def split(x):
        idx = rng.permutation(len(x))
        k = int(len(x) * 0.8)
        return x[idx[:k]], x[idx[k:]]

    ori_tr, ori_te = split(ori)
    gen_tr, gen_te = split(gen)
    model = _GRUHead(ori.shape[-1], hidden_dim, num_layers, 1, per_step=False)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(iterations):
        xr = torch.from_numpy(ori_tr[rng.choice(len(ori_tr), min(batch_size, len(ori_tr)), replace=False)])
        xf = torch.from_numpy(gen_tr[rng.choice(len(gen_tr), min(batch_size, len(gen_tr)), replace=False)])
        logits = model(torch.cat([xr, xf]))[:, 0]
        y = torch.cat([torch.ones(len(xr)), torch.zeros(len(xf))])
        opt.zero_grad()
        loss_fn(logits, y).backward()
        opt.step()
    with torch.no_grad():
        pr = model(torch.from_numpy(ori_te))[:, 0] > 0
        pf = model(torch.from_numpy(gen_te))[:, 0] > 0
    acc = (pr.sum().item() + (~pf).sum().item()) / (len(pr) + len(pf))
    return abs(0.5 - acc)
