import numpy as np
import torch
from torch import nn

def cacf_torch(x, max_lag, dim=(0, 1)):
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = []

    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, i:] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)
    
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., is_backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.is_backward = is_backward
        self.norm_foo = norm_foo

    def backward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)

class CrossCorrelationLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CrossCorrelationLoss, self).__init__(norm_foo=lambda x: torch.abs(x).sum(0), **kwargs)
        self.cross_correlation_real = cacf_torch(self.transform(x_real), 1).mean(0)[0]

    def compute(self, x_fake):
        cross_correlation_fake = cacf_torch(self.transform(x_fake), 1).mean(0)[0]
        loss = self.norm_foo(cross_correlation_fake - self.cross_correlation_real.to(x_fake.device))
        return loss / 10

def correlational_score(ori_data, generated_data, iterations=5, seed=0):
    """Correlational score (Ni et al., 2020) as computed in Diffusion-TS: mean over `iterations` random
    equal-size subsets of the CrossCorrelationLoss (L1 between lag-0 cross-correlation matrices of real and
    synthetic, /10 as in the reference code). Lower is better."""
    rng = np.random.default_rng(seed)
    x_real = torch.from_numpy(np.asarray(ori_data, np.float32))
    x_fake = torch.from_numpy(np.asarray(generated_data, np.float32))
    size = int(min(len(x_real), len(x_fake)) / iterations)
    scores = []
    for _ in range(iterations):
        r = rng.choice(len(x_real), size, replace=False)
        f = rng.choice(len(x_fake), size, replace=False)
        scores.append(CrossCorrelationLoss(x_real[r], name="CrossCorrelationLoss").compute(x_fake[f]).item())
    return float(np.mean(scores))
