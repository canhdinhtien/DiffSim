"""Predictive score (Yoon et al., NeurIPS 2019): train-on-synthetic, test-on-real (TSTR), PyTorch.

A post-hoc GRU is trained on SYNTHETIC sequences to predict the next-step vector x_{t+1} from x_{<=t}
(all features; the TimeGAN code predicts only the last feature), then evaluated on REAL sequences.
Score = MAE over all real steps and features (lower is better). Same GRU size as the discriminative score.
Inputs: equal-length arrays (N, L, F), already min-max normalized.
"""
import numpy as np
import torch
from torch import nn

from utils.discriminative_metric import _GRUHead


def predictive_score_metrics(ori_data, generated_data, iterations=2000, batch_size=128, hidden_dim=16,
                             num_layers=2, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ori, gen = np.asarray(ori_data, np.float32), np.asarray(generated_data, np.float32)
    dim = ori.shape[-1]
    model = _GRUHead(dim, hidden_dim, num_layers, dim, per_step=True)
    opt = torch.optim.Adam(model.parameters())
    for _ in range(iterations):
        x = torch.from_numpy(gen[rng.choice(len(gen), min(batch_size, len(gen)), replace=False)])
        loss = (model(x[:, :-1]) - x[:, 1:]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = torch.from_numpy(ori)
        return (model(x[:, :-1]) - x[:, 1:]).abs().mean().item()
