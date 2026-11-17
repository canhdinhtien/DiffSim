"""Context-FID (Paul et al., 2022), as implemented in Diffusion-TS: Frechet distance between TS2Vec
embeddings of real and synthetic windows, the encoder being trained on the real windows. Lower is better.
TS2Vec code is reused unmodified from src/Diffusion-TS/ts2vec (it imports itself as `Models.ts2vec`, so we alias it).
"""
import importlib
import os
import random
import sys
import types

import numpy as np
import torch
from scipy import linalg

_DTS = os.path.join(os.path.dirname(__file__), "..", "src", "Diffusion-TS")


def _ts2vec_cls():
    if "Models" not in sys.modules:
        pkg = types.ModuleType("Models")
        pkg.__path__ = [os.path.abspath(_DTS)]
        sys.modules["Models"] = pkg
    return importlib.import_module("Models.ts2vec.ts2vec").TS2Vec


def calculate_fid(act1, act2):
    """
        Calculate FID between two distribution
    """
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    sum_square_diff = np.sum((mu1 - mu2) ** 2)
    cov_mean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(cov_mean).all():  # singular product (e.g. a collapsed set): pytorch-fid's eps offset
        offset = np.eye(sigma1.shape[0]) * 1e-6
        cov_mean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = sum_square_diff + np.trace(sigma1 + sigma2 - 2.0 * cov_mean)

    return fid


def Context_FID(ori_data, generated_data, seed=0, device="cpu"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ori, gen = np.asarray(ori_data, np.float32), np.asarray(generated_data, np.float32)
    model = _ts2vec_cls()(input_dims=ori.shape[-1], device=device, batch_size=8, lr=0.001, output_dims=320,
                          max_train_length=3000)
    model.fit(ori, verbose=False)
    ori_repr = model.encode(ori, encoding_window="full_series")
    gen_repr = model.encode(gen, encoding_window="full_series")
    n = min(len(ori_repr), len(gen_repr))
    rng = np.random.default_rng(seed)
    return float(calculate_fid(ori_repr[rng.permutation(len(ori_repr))[:n]],
                               gen_repr[rng.permutation(len(gen_repr))[:n]]))
