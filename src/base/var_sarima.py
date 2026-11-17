"""M2. Pooled VAR(p) per group (vehicle_model x fleet) with day-of-week seasonal terms and a driver random effect.

Per group, on the 123-day train sequences of its drivers:
  x_{i,t} = mu_i + s_i * (season[dow_t] + h * holiday_t + z_{i,t}),  z_{i,t} = c + sum_k A_k z_{i,t-k} + e_t,
  e_t ~ N(0, Sigma).
- random_effect: "mean_scale" (driver mean mu_i and scale s_i), "mean" (mu_i, pooled within-driver scale),
  "off" (group mean and group std for everyone). Driver effects [mu_i, log s_i] ~ fitted multivariate Gaussian.
- VAR coefficients by pooled OLS across drivers (one model per group, never per driver).
- Sigma (and the random-effect covariance) shrunk toward their diagonal by `shrink`.
SARIMA is not used (VAR(p) with dow dummies chosen; p=7 covers the seasonal AR lag). Post-processing: clip, round.
"""
import numpy as np

from src.base.common import DOW, HOL, Generator, days


def _shrink(S, lam):
    return (1 - lam) * S + lam * np.diag(np.diag(S))


def fit_seasonal(Z, seasonal):
    """Day-of-week means (7, F) and holiday shift (F,) of Z (n, T, F) over the full calendar."""
    if not seasonal:
        return np.zeros((7, Z.shape[-1])), np.zeros(Z.shape[-1])
    season = np.stack([Z[:, (DOW == k) & (HOL == 0)].mean((0, 1)) for k in range(7)])
    hol = (Z[:, HOL == 1] - season[DOW[HOL == 1]]).mean((0, 1))
    return season, hol


def fit_var(Z, p, ridge=1e-6):
    """Pooled OLS VAR(p) with intercept on Z (n, T, F). Returns c (F,), A (p, F, F), residuals (N, F)."""
    n, T, F = Z.shape
    Y = Z[:, p:].reshape(-1, F)
    Xlag = np.concatenate([np.ones((n, T - p, 1))] + [Z[:, p - k:T - k] for k in range(1, p + 1)], -1).reshape(len(Y), -1)
    B = np.linalg.solve(Xlag.T @ Xlag + ridge * np.eye(Xlag.shape[1]), Xlag.T @ Y)
    A = B[1:].reshape(p, F, F).transpose(0, 2, 1)  # A[k] maps z_{t-k-1} -> z_t
    return B[0], A, Y - Xlag @ B


def simulate_var(c, A, chol, n, L, rng, burn=100):
    p, F = A.shape[0], len(c)
    z = np.zeros((n, burn + L + p, F))
    eps = rng.standard_normal((n, burn + L + p, F)) @ chol.T
    for t in range(p, burn + L + p):
        z[:, t] = c + eps[:, t] + sum(z[:, t - k - 1] @ A[k].T for k in range(p))
    return z[:, -L:]


class PooledVAR(Generator):
    def __init__(self, p=1, seasonal=True, shrink=0.0, random_effect="mean_scale"):
        assert random_effect in ("off", "mean", "mean_scale")
        self.p, self.seasonal, self.shrink, self.random_effect = p, seasonal, shrink, random_effect

    def _fit(self):
        self.models = []
        for g in range(self.n_groups):
            Xg = self.X[self.groups == g]
            mu, sd = Xg.mean(1), Xg.std(1) + 1e-3
            if self.random_effect == "off":
                mu, sd = np.broadcast_to(Xg.mean((0, 1)), mu.shape), np.broadcast_to(Xg.std((0, 1)), sd.shape)
            elif self.random_effect == "mean":
                sd = np.broadcast_to(np.sqrt((Xg.var(1)).mean(0)) + 1e-3, sd.shape)
            Z = (Xg - mu[:, None]) / sd[:, None]
            season, hol = fit_seasonal(Z, self.seasonal)
            Z = Z - season[DOW] - HOL[:, None] * hol
            c, A, resid = fit_var(Z, self.p)
            Sigma = _shrink(np.cov(resid, rowvar=False), self.shrink)
            theta = np.concatenate([mu, np.log(sd)], 1)
            C = _shrink(np.cov(theta, rowvar=False), self.shrink) + 1e-9 * np.eye(theta.shape[1])
            rho = np.abs(np.linalg.eigvals(self._companion(A))).max()
            self.models.append(dict(c=c, A=A, chol=np.linalg.cholesky(Sigma), season=season, hol=hol,
                                    theta_mean=theta.mean(0), theta_cov=C, rho=rho, mu=mu[0], sd=sd[0]))

    @staticmethod
    def _companion(A):
        p, F, _ = A.shape
        M = np.zeros((p * F, p * F))
        M[:F] = np.concatenate(list(A), 1)
        M[F:, :-F] = np.eye((p - 1) * F)
        return M

    def _sample(self, group, start, rng):
        F = self.X.shape[-1]
        out = np.empty((len(group), self.L, F))
        dd = days(start, self.L)
        for g in np.unique(group):
            m, idx = self.models[g], np.flatnonzero(group == g)
            if self.random_effect == "off":
                mu, sd = np.tile(m["mu"], (len(idx), 1)), np.tile(m["sd"], (len(idx), 1))
            else:
                th = rng.multivariate_normal(m["theta_mean"], m["theta_cov"], len(idx))
                mu = th[:, :F]
                sd = np.exp(th[:, F:]) if self.random_effect == "mean_scale" else np.tile(m["sd"], (len(idx), 1))
            z = simulate_var(m["c"], m["A"], m["chol"], len(idx), self.L, rng)
            z = z + m["season"][DOW[dd[idx]]] + HOL[dd[idx]][..., None] * m["hol"]
            out[idx] = mu[:, None] + sd[:, None] * z
        return out
