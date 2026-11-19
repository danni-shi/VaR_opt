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


def simulate_poisson_process_discrete(N, intensity=0.1):
    """
    Simulates a Poisson process on a discrete time grid.
    
    Parameters:
    - N: Number of time steps
    - intensity: Rate parameter (lambda) of the Poisson process
    - tau: Time step size
    - seed: Random seed for reproducibility
    
    Returns:
    - events: Binary array indicating events at each time step
    - event_times: Indices where events occurred
    """
    
    # Probability of event in each time step
    prob_event = intensity 
    
    # Generate events (Bernoulli trials)
    events = np.random.binomial(1, prob_event, size=N)
    event_times = np.where(events == 1)[0]

        
    return events, event_times


def data_syn_tim(params, sigma, tau, n_trades, N, method='gaps', min_gap=10, seed=42):
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

    if method == 'gaps':
    # Sanity-check for enough room
        if N < min_gap * (n_trades - 1) + n_trades:
            raise ValueError(
                f"N={N} too small for {n_trades=} with {min_gap=}. "
                f"Need at least {min_gap * (n_trades - 1) + n_trades} steps."
            )

        # Draw, sort, then offset
        u = rng.choice(N - min_gap * (n_trades), n_trades, replace=False)
        u.sort()
        poission_pos = u + min_gap * np.arange(n_trades)
        v_sim = np.zeros(N)
        v_sim[poission_pos] = np.clip(
        1e6 * (0.5+np.random.exponential(scale=1, size=len(poission_pos))), 
                         a_min=None, a_max=8e6)

    elif method == 'poisson':
        _, poission_pos = simulate_poisson_process_discrete(N, intensity=n_trades/N, seed=seed)
        v_sim = np.zeros(N)
        v_sim[poission_pos] = np.clip(
            1e6 * (0.5+np.random.exponential(scale=1, size=len(poission_pos))), 
                            a_min=None, a_max=8e6)  * np.random.choice([-1, 1], size=len(poission_pos))
        
    elif method == 'gaussian':
        v_sim = np.random.normal(loc=0, scale=1e5, size=N)
    
    elif method == 'poisson_gaussian':
        _, poission_pos = simulate_poisson_process_discrete(N, intensity=n_trades/N, seed=seed)
        v_sim = np.zeros(N)
        v_sim[poission_pos] = 0.5 *np.clip(
            1e6 * (0.5+np.random.exponential(scale=1, size=len(poission_pos))), 
                            a_min=None, a_max=8e6) * np.random.choice([-1, 1], size=len(poission_pos))
        v_sim += 0.5 * np.random.normal(loc=0, scale=1.5e4, size=N)
    elif method == 'gaps_poisson':
        # Create base pattern with trades every 100 steps
        v_sim = np.zeros(N)
        open = np.arange(0, N, int(1/tau))  # Every 100 elements
        
        # Add deterministic trades at regular intervals
        v_sim[open] = np.clip(
            1e6 * (0.5 + np.random.exponential(scale=1, size=len(open))), 
            a_min=None, a_max=8e6) * np.random.choice([-1, 1], size=len(open))
        
        # Add Poisson trades
        _, poission_pos = simulate_poisson_process_discrete(N, intensity=5*tau, seed=seed)

        v_sim[poission_pos] += np.clip(
            1e5 * (0.5+np.random.exponential(scale=1, size=len(poission_pos))), 
                            a_min=None, a_max=8e6)  * np.random.choice([-1, 1], size=len(poission_pos))
        
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    
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



def lognormal_params(mean_X, std_X):
    """
    Computes the parameters for a log-normal distribution.

    Parameters:
    - mean_X: Mean of the lognormal distribution.
    - std_X: Standard deviation of the lognormal distribution.

    Returns:
    - mu_log: Mean of the underlying normal distribution in log space
    - sigma_log: Standard deviation of the underlying normal distribution in the log-space.
    """
    sigma_log = np.sqrt(np.log(1 + (std_X / mean_X)**2))
    mu_log = np.log(mean_X) - 0.5 * sigma_log**2
    assert np.isclose(np.exp(mu_log + 0.5 * sigma_log**2), mean_X), "Mean calculation mismatch"
    assert np.isclose((np.exp(sigma_log**2) - 1) * np.exp(2 * mu_log + sigma_log**2), std_X**2), "Std calculation mismatch"
    return mu_log, sigma_log


if __name__ == "__main__":
    params = {"kappa": 5e-5, "rho": 2.231, "gamma": 5e-5}
    sigma = 0.01
    tau = 0.1
    n_days = 3
    S, v_sim = data_syn_tim_by_day(params, sigma, tau, n_days, method='gaps_poisson', seed=42)
    
