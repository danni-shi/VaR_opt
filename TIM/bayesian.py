import numpy as np
import os
from datetime import datetime
import time
import pymc as pm
import pytensor.tensor as at
from pytensor.scan import scan
import arviz as az
import pytensor
import pickle
from multiprocessing import Pool
from data_syn import data_syn_tim, lognormal_params

# ─── Configure PyTensor ───
pytensor.config.mode = "NUMBA"
pytensor.config.cxx = ""

# ─── Function for Data Simulation ───
def simulate_data(params, sigma, tau, n_trades, min_gap, seed):
    """
    Simulates trading data based on the given parameters.
    """
    N = int(1.5 * min_gap * n_trades + n_trades)
    S, v_sim = data_syn_tim(params, sigma, tau, n_trades, N, min_gap, seed)
    return S, v_sim

# ─── Function for Bayesian Modeling ───
def inference_tim(S, v_sim, tau, sigma_true, outdir, n_trades, min_gap, seed):
    """
    Runs the Bayesian inference model using PyMC.
    """
    # Prepare data
    v_data = v_sim         # shape (T,)
    S_data = S             # shape (T+1,)
    dS_data = S_data[1:] - S_data[:-1]  # increments shape (T,)

    # Define priors
    k_mu, k_sigma = lognormal_params(1.5e-5, 1e-5)  # for θ
    g_mu, g_sigma = lognormal_params(1.5e-5, 1e-5)  # for α∈[0,1]
    r_mu, r_sigma = lognormal_params(3, 2)          # for rho

    with pm.Model() as model:
        # Priors
        rho = pm.LogNormal("rho", mu=r_mu, sigma=r_sigma)
        kappa = pm.LogNormal("kappa", mu=k_mu, sigma=k_sigma)
        gamma = pm.LogNormal("gamma", mu=g_mu, sigma=g_sigma)

        # Data inputs
        v = at.constant(v_data)    # (T,)
        S_obs = at.constant(dS_data)   # (T,)

        # Exponential-impact state via scan
        def step(v_k, i_prev, kappa, rho, tau):
            return i_prev + (kappa * v_k - rho * i_prev) * tau

        i_seq, _ = scan(
            fn=step,
            sequences=[v],
            outputs_info=[at.zeros(())],
            non_sequences=[kappa, rho, tau]
        )  # shape (T,)

        # Shift to get i_{k-1}
        i_prev = at.concatenate([at.zeros((1,)), i_seq[:-1]])  # (T,)

        # Build the drift on price increments
        drift_perm = gamma * tau * v       # permanent part
        drift_trans = (i_seq - i_prev)  # transient part
        mu_inc = -drift_perm - drift_trans

        # Likelihood
        sigma_obs = sigma_true * at.sqrt(tau)
        pm.Normal("dS", mu=mu_inc, sigma=sigma_obs, observed=S_obs)

        # Inference
        trace = pm.sample(
            draws=1000,
            tune=4000,
            cores=1,
            chains=10,
            target_accept=0.99,
            return_inferencedata=True
        )

    return trace

def inference_tim_reparam(S, v_sim, tau, sigma_true, outdir, n_trades, min_gap, seed):
    """
    Runs the Bayesian inference model using PyMC with reparameterization.
    """
    # Prepare data
    v_data = v_sim         # shape (T,)
    S_data = S             # shape (T+1,)
    dS_data = S_data[1:] - S_data[:-1]  # increments shape (T,)

    # Define priors
    t_mu, t_sigma = lognormal_params(3e-5, 2e-5)  # for θ
    a_alpha, b_alpha = 4, 4                        # for α∈[0,1]
    r_mu, r_sigma = lognormal_params(3, 2)        # for rho

    with pm.Model() as model:
        # Priors on reparameterized space
        theta = pm.LogNormal("theta", mu=t_mu, sigma=t_sigma)
        alpha = pm.Beta("alpha", alpha=a_alpha, beta=b_alpha)
        rho = pm.LogNormal("rho", mu=r_mu, sigma=r_sigma)

        # Deterministic transforms back to kappa, gamma
        kappa = pm.Deterministic("kappa", theta * alpha)
        gamma = pm.Deterministic("gamma", theta * (1 - alpha))

        # Data inputs
        v = at.constant(v_data)    # (T,)
        S_obs = at.constant(dS_data)   # (T,)

        # Exponential-impact state via scan
        def step(v_k, i_prev, kappa, rho, tau):
            return i_prev + (kappa * v_k - rho * i_prev) * tau

        i_seq, _ = scan(
            fn=step,
            sequences=[v],
            outputs_info=[at.zeros(())],
            non_sequences=[kappa, rho, tau]
        )  # shape (T,)

        # Shift to get i_{k-1}
        i_prev = at.concatenate([at.zeros((1,)), i_seq[:-1]])  # (T,)

        # Build the drift on price increments
        drift_perm = gamma * tau * v       # permanent part
        drift_trans = (i_seq - i_prev)  # transient part
        mu_inc      = -drift_perm - drift_trans

        # — Likelihood — 
        sigma_obs = sigma_true * at.sqrt(tau)
        pm.Normal("dS", mu=mu_inc, sigma=sigma_obs, observed=S_obs)

        # — Inference — 
        trace = pm.sample(
            draws=1000,
            tune=4000,
            cores = 1,
            chains = 10,
            target_accept=0.99, 
            return_inferencedata=True
        )
        return trace
    
# ─── Worker Function for Parallel Execution ───
def process_simulation(args):
    """
    Worker function to simulate data and perform Bayesian inference for a single combination of n_trades and run.
    """
    n_trades, run, params, sigma, tau, min_gap, seed, outdir, datadir = args
    S, v_sim = simulate_data(params, sigma, tau, n_trades, min_gap, seed)

    # Save simulated data to pickle file
    output_file = os.path.join(datadir, f"sim_data_{n_trades}_gap_{min_gap}_seed_{seed}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump((S, v_sim), f)

    # Perform Bayesian inference
    # trace = inference_tim(S, v_sim, tau, sigma, outdir, n_trades, min_gap, seed)
    trace = inference_tim_reparam(S, v_sim, tau, sigma, outdir, n_trades, min_gap, seed)

    # Save results
    trace.to_netcdf(os.path.join(outdir, f"trace_{n_trades}_gap_{min_gap}_seed_{seed}.nc"))

# ─── Main Execution ───
def main():
    # Start timer
    start_time = time.time()
    impact_model = 'TIM'
    # Create output directories
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(f"{impact_model}_MCMC_Results", now)
    datadir = os.path.join(f"{impact_model}_simulated_data", now)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    # Trade list and parameters
    trade_list = [500, 800, 1100, 1400]

    num_runs = 15
    params = {
        "kappa": 1e-5,   # transient impact scale
        "rho": 2.231,    # decay rate
        "gamma": 1e-5    # permanent impact scale
    }
    sigma = 0.45   # volatility
    tau = 0.01
    min_gap = 100
    seed = 115

    # Prepare arguments for parallel execution
    args_list = [
        (n_trades, run, params, sigma, tau, min_gap, seed+run, outdir, datadir)
        for n_trades in trade_list
        for run in range(num_runs)
    ]

    # Parallelize the double loop
    with Pool() as pool:
        pool.map(process_simulation, args_list)

    # End timer
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Elapsed time: {run_time:.6f} seconds")

# Run the script
if __name__ == "__main__":
    main()