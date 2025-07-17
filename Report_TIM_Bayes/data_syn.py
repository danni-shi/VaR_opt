import numpy as np
import matplotlib.pyplot as plt

def data_syn(kappa, beta, gamma, sigma, tau, n_trades, N, min_gap=10, seed=42):
    rng = np.random.default_rng(seed)

    # sanity‐check for enough room
    if N < min_gap*(n_trades-1) + n_trades:
        raise ValueError(
            f"N={N} too small for {n_trades=} with {min_gap=}. "
            f"Need at least {min_gap*(n_trades-1)+n_trades} steps."
        )

    # draw, sort, then offset
    u = rng.choice(N - min_gap*(n_trades), n_trades, replace=False)
    u.sort()
    pos = u + min_gap * np.arange(n_trades)

    v_sim = np.zeros(N)
    v_sim[pos] = rng.uniform(1e3, 1e5, size=n_trades)

    I = np.zeros(N+1)
    S = np.zeros(N+1)
    S[0] = 50.0
    dW = rng.normal(0, np.sqrt(tau), size=N)

    for k in range(1, N+1):
        I[k] = I[k-1] + (kappa * v_sim[k-1] - beta * I[k-1]) * tau
        S[k] = (
            S[k-1]
            - (gamma * v_sim[k-1] + (kappa * v_sim[k-1] - beta * I[k-1])) * tau
            + sigma * dW[k-1]
        )

    return S, v_sim




def lognormal_params(mu_X, sigma_X):
    sigma_log = np.sqrt(np.log(1 + (sigma_X / mu_X)**2))
    mu_log = np.log(mu_X) - 0.5 * sigma_log**2
    return mu_log, sigma_log