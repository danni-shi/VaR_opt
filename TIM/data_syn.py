import numpy as np
import matplotlib.pyplot as plt

def step(v_k, i_prev, kappa, rho, tau):
    return i_prev + (kappa * v_k - rho * i_prev) * tau

def fn_dI(v_k, i_prev, kappa, rho, tau):
    """
    Computes the change in impact state given the current volume, previous impact state,
    and model parameters.
    
    Parameters:
    - v_k: Current trade volume.
    - i_prev: Previous impact state.
    - kappa: Sensitivity of impact to trade volume.
    - rho: Decay rate of impact.
    - tau: Time step.
    
    Returns:
    - Change in impact state.
    """
    return (kappa * v_k - rho * i_prev) * tau

def fn_dS(gamma, v, tau, dI):
    return -(gamma * v * tau + dI)

def data_syn_tim(params, sigma, tau, n_trades, N, min_gap=10, seed=42):
    """
    Simulates trading data based on the given parameters.
    
    Parameters:
    - params: Dictionary containing kappa, rho, and gamma.
    - sigma: Volatility.
    - tau: Time step.
    - n_trades: Number of trades to simulate.
    - N: Total number of time steps.
    - min_gap: Minimum gap between trades.
    - seed: Random seed for reproducibility.
    Returns:
    - S: Simulated price series.
    - v_sim: Simulated trade volumes.
    """
    rng = np.random.default_rng(seed)

    # Extract parameters from the dictionary
    kappa = params["kappa"]
    rho = params["rho"]
    gamma = params["gamma"]

    # Sanity-check for enough room
    if N < min_gap * (n_trades - 1) + n_trades:
        raise ValueError(
            f"N={N} too small for {n_trades=} with {min_gap=}. "
            f"Need at least {min_gap * (n_trades - 1) + n_trades} steps."
        )

    # Draw, sort, then offset
    u = rng.choice(N - min_gap * (n_trades), n_trades, replace=False)
    u.sort()
    pos = u + min_gap * np.arange(n_trades)

    v_sim = np.zeros(N)
    # v_sim[pos] = rng.uniform(1e3, 1e5, size=n_trades)
    v_sim[pos] = np.clip(
        1e6 * (0.5+np.random.exponential(scale=1, size=n_trades)), 
                         a_min=None, a_max=8e6)
    I = np.zeros(N + 1)
    S = np.zeros(N + 1)
    S[0] = 50.0
    dW = rng.normal(0, np.sqrt(tau), size=N)

    for k in range(1, N + 1):
        dI = fn_dI(v_sim[k - 1], I[k - 1], kappa, rho, tau)  # v_sim[k-1] is actually n_k
        I[k] = I[k - 1] + dI  # forward Euler
        I_k = step(v_sim[k - 1], I[k - 1], kappa, rho, tau)
        assert I[k] == I_k, f"Mismatch at step {k}: {I[k]} != {I_k}"
        S[k] = (
            S[k - 1]
            + fn_dS(gamma, v_sim[k - 1], tau, dI)
            + sigma * dW[k - 1]
        )

    return S, v_sim



def lognormal_params(mu_X, sigma_X):
    """
    Computes the parameters for a log-normal distribution.

    Parameters:
    - mu_X: Mean of the lognormal distribution.
    - sigma_X: Standard deviation of the lognormal distribution.

    Returns:
    - mu_log: Mean of the underlying normal distribution in log space
    - sigma_log: Standard deviation of the underlying normal distribution in the log-space.
    """
    sigma_log = np.sqrt(np.log(1 + (sigma_X / mu_X)**2))
    mu_log = np.log(mu_X) - 0.5 * sigma_log**2
    return mu_log, sigma_log