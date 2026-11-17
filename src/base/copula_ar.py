"""M3. Gaussian copula + AR marginal dynamics, fitted per group (vehicle_model x fleet) on pooled train drivers.

1. Marginal per (group, feature): empirical CDF, or parametric (negative binomial by moments for counts,
   Poisson if under-dispersed; lognormal for online_time_h). Probability integral transform to normal scores
   z = Phi^-1(u); ties of discrete values: u = F(x-) + V (F(x) - F(x-)), V ~ U(0,1) ("randomized") or V = 0.5
   ("midpoint").
2. Dynamics in z-space: z_{i,t} = a_i + season[dow_t] + h*holiday_t + r_{i,t}, a_i ~ N(m, C) driver random effect,
   r_{i,t,f} = c_f + sum_k phi_{f,k} r_{i,t-k,f} + e_{i,t,f} (per-feature AR(p), pooled OLS).
3. Gaussian copula across features at each step: e_t = sigma * N(0, R), R the innovation correlation shrunk
   toward I by `shrink`.
Sampling inverts the marginal: x = F^-1(Phi(z)).
"""
import numpy as np
from scipy import stats

from src.base.common import DOW, HOL, I_ONLINE, Generator, days
from src.base.var_sarima import fit_seasonal

EPS = 1e-6


class _Marginal:
    def __init__(self, x, family, discrete):
        self.discrete, self.family = discrete, family
        self.v = np.sort(x)
        if family == "parametric":
            if discrete:
                m, var = x.mean(), x.var()
                self.dist = stats.poisson(m) if var <= m * 1.001 else stats.nbinom(m * m / (var - m), m / var)
            else:
                s, loc, scale = stats.lognorm.fit(x, floc=0)
                self.dist = stats.lognorm(s, loc, scale)

    def cdf_pair(self, x):
        """(F(x-), F(x)) for the PIT."""
        if self.family == "empirical":
            n = len(self.v)
            return np.searchsorted(self.v, x, "left") / n, np.searchsorted(self.v, x, "right") / n
        if self.discrete:
            return self.dist.cdf(x - 1), self.dist.cdf(x)
        F = self.dist.cdf(x)
        return F, F

    def ppf(self, u):
        if self.family == "parametric":
            return self.dist.ppf(u)
        return np.quantile(self.v, u, method="inverted_cdf" if self.discrete else "linear")


class GaussianCopulaAR(Generator):
    def __init__(self, p=1, family="empirical", shrink=0.0, ties="randomized", seed=0):
        assert family in ("empirical", "parametric") and ties in ("randomized", "midpoint")
        self.p, self.family, self.shrink, self.ties, self.seed = p, family, shrink, ties, seed

    def _fit(self):
        rng = np.random.default_rng(self.seed)
        F = self.X.shape[-1]
        self.models = []
        for g in range(self.n_groups):
            Xg = self.X[self.groups == g]
            margs, Z = [], np.empty_like(Xg)
            for f in range(F):
                m = _Marginal(Xg[..., f].ravel(), self.family, discrete=f != I_ONLINE)
                lo, hi = m.cdf_pair(Xg[..., f])
                v = rng.uniform(size=lo.shape) if self.ties == "randomized" else 0.5
                Z[..., f] = stats.norm.ppf(np.clip(lo + v * (hi - lo), EPS, 1 - EPS))
                margs.append(m)
            a = Z.mean(1)
            season, hol = fit_seasonal(Z - a[:, None], True)
            R = Z - a[:, None] - season[DOW] - HOL[:, None] * hol
            T = R.shape[1]
            coef, resid = [], []
            for f in range(F):
                y = R[:, self.p:, f].ravel()
                X = np.stack([np.ones_like(y)] + [R[:, self.p - k:T - k, f].ravel() for k in range(1, self.p + 1)], 1)
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                coef.append(b)
                resid.append(y - X @ b)
            resid = np.stack(resid, 1)
            sigma = resid.std(0)
            corr = (1 - self.shrink) * np.corrcoef(resid, rowvar=False) + self.shrink * np.eye(F)
            self.models.append(dict(margs=margs, a_mean=a.mean(0), a_cov=np.cov(a, rowvar=False) + 1e-9 * np.eye(F),
                                    season=season, hol=hol, coef=np.stack(coef), sigma=sigma,
                                    chol=np.linalg.cholesky(corr)))

    def _sample(self, group, start, rng, burn=100):
        F = self.X.shape[-1]
        out = np.empty((len(group), self.L, F))
        dd = days(start, self.L)
        for g in np.unique(group):
            m, idx = self.models[g], np.flatnonzero(group == g)
            n, Tt = len(idx), burn + self.L + self.p
            e = (rng.standard_normal((n, Tt, F)) @ m["chol"].T) * m["sigma"]
            r = np.zeros((n, Tt, F))
            c, phi = m["coef"][:, 0], m["coef"][:, 1:]
            for t in range(self.p, Tt):
                r[:, t] = c + e[:, t] + sum(phi[:, k] * r[:, t - k - 1] for k in range(self.p))
            a = rng.multivariate_normal(m["a_mean"], m["a_cov"], n)
            z = a[:, None] + m["season"][DOW[dd[idx]]] + HOL[dd[idx]][..., None] * m["hol"] + r[:, -self.L:]
            u = np.clip(stats.norm.cdf(z), EPS, 1 - EPS)
            for f in range(F):
                out[idx, :, f] = m["margs"][f].ppf(u[..., f])
        return out
