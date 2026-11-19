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
import argparse
import json
from multiprocessing import Pool
from data_syn import data_syn_tim, lognormal_params, fn_dI, step
from functools import partial

# ─── Configure PyTensor ───
pytensor.config.mode = "NUMBA"
pytensor.config.cxx = ""




### ─── Prior Parameters ───
T_MEAN_PRIOR, T_STD_PRIOR = 5e-4, 2e-4
ALPHA_A_PRIOR, ALPHA_B_PRIOR = 2, 1
RHO_MEAN_PRIOR, RHO_STD_PRIOR = 8, 2
    
# ─── Function for Data Simulation ───
def simulate_data(params, sigma, tau, n_trades, method, min_gap, seed):
    """
    Simulates trading data based on the given parameters.
    """
    if method == 'gaps':
        N = int(1.5 * min_gap * n_trades + n_trades)
    elif 'poisson' in method:
        N = int(n_trades/tau)  # assume intensity of poisson trade process is 1/100
    else: 
        N = n_trades
    S, v_sim = data_syn_tim(params, sigma, tau, n_trades, N, method, min_gap, seed)
    return S, v_sim

# ─── Function for Bayesian Modeling ───
# def inference_tim(S, v_sim, tau, sigma_true):
#     """
#     Runs the Bayesian inference model using PyMC.
#     """
#     # Prepare data
#     v_data = v_sim         # shape (T,)
#     S_data = S             # shape (T+1,)
#     dS_data = S_data[1:] - S_data[:-1]  # increments shape (T,)

#     # Define priors
#     k_mu, k_sigma = lognormal_params(1.5e-5, 1e-5)  # for θ
#     g_mu, g_sigma = lognormal_params(1.5e-5, 1e-5)  # for α∈[0,1]
#     r_mu, r_sigma = lognormal_params(3, 2)          # for rho

#     with pm.Model() as model:
#         # Priors
#         rho = pm.LogNormal("rho", mu=r_mu, sigma=r_sigma)
#         kappa = pm.LogNormal("kappa", mu=k_mu, sigma=k_sigma)
#         gamma = pm.LogNormal("gamma", mu=g_mu, sigma=g_sigma)

#         # Data inputs
#         v = at.constant(v_data)    # (T,)
#         S_obs = at.constant(dS_data)   # (T,)

#         # Exponential-impact state via scan
#         def step(v_k, i_prev, kappa, rho, tau):
#             return i_prev + (kappa * v_k - rho * i_prev) * tau

#         i_seq, _ = scan(
#             fn=step,
#             sequences=[v],
#             outputs_info=[at.zeros(())],
#             non_sequences=[kappa, rho, tau]
#         )  # shape (T,)

#         # Shift to get i_{k-1}
#         i_prev = at.concatenate([at.zeros((1,)), i_seq[:-1]])  # (T,)

#         # Build the drift on price increments
#         drift_perm = gamma * tau * v       # permanent part
#         drift_trans = (i_seq - i_prev)  # transient part
#         mu_inc = -drift_perm - drift_trans

#         # Likelihood
#         sigma_obs = sigma_true * at.sqrt(tau)
#         pm.Normal("dS", mu=mu_inc, sigma=sigma_obs, observed=S_obs)

#         # Inference
#         trace = pm.sample(
#             draws=1000,
#             tune=4000,
#             cores=1,
#             chains=10,
#             target_accept=0.99,
#             return_inferencedata=True
#         )

#     return trace

# def get_transient_impact_pm(v, kappa, gamma, rho, tau):
#     # Exponential-impact state via scan
#     def step(v_k, i_prev, kappa, rho, tau):
#         return i_prev + (kappa * v_k - rho * i_prev) * tau
#     i_seq, _ = scan(
#     fn=step,
#     sequences=[v],
#     outputs_info=[at.zeros(())],
#     non_sequences=[kappa, rho, tau]
#     )  # shape (T,)

#     # Shift to get i_{k-1}
#     i_prev = at.concatenate([at.zeros((1,)), i_seq[:-1]])  # (T,)

#     drift_trans = (i_seq - i_prev)  # transient part
#     return drift_trans

# def inference_tim_reparam(S, v_sim, tau, sigma_true):
#     """
#     Runs the Bayesian inference model using PyMC with reparameterization.
#     """
#     # Prepare data
#     v_data = v_sim         # shape (T,)
#     S_data = S             # shape (T+1,)
#     dS_data = S_data[1:] - S_data[:-1]  # increments shape (T,)

#     # Define priors
#     t_mu, t_sigma = lognormal_params(T_MEAN_PRIOR, T_STD_PRIOR)  # for θ
#     a_alpha, b_alpha = ALPHA_A_PRIOR, ALPHA_B_PRIOR               # for α∈[0,1]
#     r_mu, r_sigma = lognormal_params(RHO_MEAN_PRIOR, RHO_STD_PRIOR)  # for rho

#     with pm.Model() as model:
#         # Priors on reparameterized space
#         theta = pm.LogNormal("theta", mu=t_mu, sigma=t_sigma)
#         alpha = pm.Beta("alpha", alpha=a_alpha, beta=b_alpha)
#         rho = pm.LogNormal("rho", mu=r_mu, sigma=r_sigma)

#         # Deterministic transforms back to kappa, gamma
#         kappa = pm.Deterministic("kappa", theta * alpha)
#         gamma = pm.Deterministic("gamma", theta * (1 - alpha))

#         # Data inputs
#         v = at.constant(v_data)    # (T,)
#         dS = at.constant(dS_data)   # (T,)

#         i_seq, _ = scan(
#             fn=step,
#             sequences=[v],
#             outputs_info=[at.zeros(())],
#             non_sequences=[kappa, rho, tau]
#         )  # shape (T,)

#         # Shift to get i_{k-1}
#         i_prev = at.concatenate([at.zeros((1,)), i_seq[:-1]])  # (T,)

#         # Build the drift on price increments
#         drift_perm = gamma * tau * v       # permanent part
#         drift_trans = (i_seq - i_prev)  # transient part 
#         mu_inc      = -drift_perm - drift_trans

#         # — Likelihood — 
#         sigma_obs = sigma_true * at.sqrt(tau)
#         pm.Normal("dS", mu=mu_inc, sigma=sigma_obs, observed=dS)

#         # — Inference — 
#         trace = pm.sample(
#             draws=1000,
#             tune=3000,
#             cores = 1,
#             chains = 8,
#             target_accept=0.99, 
#             return_inferencedata=True
#         )
#         return trace
        
def inference_tim_reparam_online(ntrades_list, S, v_sim, sigma, tau, min_gap, seed, outdir, prior_params, **mcmc_kwargs):
    """
    Runs the Bayesian inference model using PyMC with reparameterization for online learning.
    
    Parameters:
    -----------
    S : array_like
        Price series (shape T+1)
    v_sim : array_like
        Volume series (shape T)
    tau : float
        Time step
    sigma_true : float
        True volatility
    data_points_list : list
        List of data points to use for each stage (e.g., [1000, 2000, 3000])
    outdir : str, optional
        Output directory to save traces
    identifier : str, optional
        Identifier for naming output files
        
    Returns:
    --------
    traces : dict
        Dictionary with keys as data points and values as MCMC traces
    """
    # Perform online Bayesian inference
    trade_times = np.nonzero(v_sim)[0]
    cutoff_points = np.concatenate((trade_times , [len(v_sim)])) # index to truncate data after a number of trades are observed
    data_points_list = cutoff_points[list(ntrades_list)]
    
    # Prepare full data
    v_data = v_sim         # shape (T,)
    S_data = S             # shape (T+1,)
    dS_data = S_data[1:] - S_data[:-1]  # increments shape (T,)
    
    # Define priors (same for all stages)
    t_mu, t_sigma = lognormal_params(prior_params["t_mean_prior"], prior_params["t_std_prior"])  # for θ
    a_alpha, b_alpha = prior_params["alpha_a_prior"], prior_params["alpha_b_prior"]                        # for α∈[0,1]
    r_mu, r_sigma = lognormal_params(prior_params["rho_mean_prior"], prior_params["rho_std_prior"])        # for rho

    for ntrades, n_points in zip(ntrades_list, data_points_list):
        print(f"Running MCMC with first {ntrades} trades ({n_points} data points)...")
        
        # Truncate data to first n_points
        if n_points > len(v_data):
            print(f"Warning: Requested {n_points} points but only {len(v_data)} available. Using all data.")
            n_points = len(v_data)
        
        v_truncated = v_data[:n_points]
        dS_truncated = dS_data[:n_points]
        
        with pm.Model() as model:
            # Priors on reparameterized space
            theta = pm.LogNormal("theta", mu=t_mu, sigma=t_sigma)
            alpha = pm.Beta("alpha", alpha=a_alpha, beta=b_alpha)
            rho = pm.LogNormal("rho", mu=r_mu, sigma=r_sigma)

            # Deterministic transforms back to kappa, gamma
            kappa = pm.Deterministic("kappa", theta * alpha)
            gamma = pm.Deterministic("gamma", theta * (1 - alpha))

            # Data inputs (truncated)
            v = at.constant(v_truncated)    # (n_points,)
            dS = at.constant(dS_truncated)   # (n_points,)

            i_seq, _ = scan(
                fn=step,
                sequences=[v],
                outputs_info=[at.zeros(())],
                non_sequences=[kappa, rho, tau]
            )  # shape (n_points,)

            # Shift to get i_{k-1}
            i_prev = at.concatenate([at.zeros((1,)), i_seq[:-1]])  # (n_points,)

            # Build the drift on price increments
            drift_perm = gamma * tau * v       # permanent part
            drift_trans = (i_seq - i_prev)  # transient part 
            mu_inc = -drift_perm - drift_trans

            # Likelihood
            sigma_obs = sigma * at.sqrt(tau)
            pm.Normal("dS", mu=mu_inc, sigma=sigma_obs, observed=dS)

            # Inference
            trace = pm.sample(
                **mcmc_kwargs
            )
            
        mcmc_file = f"trace_{ntrades}_gap_{min_gap}_seed_{seed}.nc"
        trace.to_netcdf(os.path.join(outdir, mcmc_file))
    
 
# ─── Worker Function for Parallel Execution ───
# def process_simulation(ntrades, params, sigma, tau, min_gap, seed, outdir, datadir):
#     """
#     Worker function to simulate data and perform Bayesian inference for a single combination of n_trades and run.
#     Regenerate data for each inference
#     """
#     #n_trades, run, params, sigma, tau, min_gap, seed, outdir, datadir = args
#     S, v_sim = simulate_data(params, sigma, tau, ntrades, min_gap, seed)

#     # Save simulated data to pickle file
#     output_file = os.path.join(datadir, f"sim_data_{ntrades}_gap_{min_gap}_seed_{seed}.pkl")
#     with open(output_file, 'wb') as f:
#         pickle.dump((S, v_sim), f)

#     # Perform Bayesian inference
#     # trace = inference_tim(S, v_sim, tau, sigma, outdir, n_trades, min_gap, seed)
#     trace = inference_tim_reparam(S, v_sim, tau, sigma)

#     # Save results
#     trace.to_netcdf(os.path.join(outdir, f"trace_{ntrades}_gap_{min_gap}_seed_{seed}.nc"))


def process_simulation_online(ntrades_list, params, sigma, tau, data_method, min_gap, seed, outdir, datadir, prior_params, **mcmc_kwargs):
    """
    Worker function for online learning with multiple MCMC posteriors.
    """
    np.random.seed(seed)
    ntrades = max(ntrades_list)
    # S_new, v_sim_new = simulate_data(params, sigma, tau, 2000, min_gap, data_method,  seed)
    mcmc_file = os.path.join(datadir, f"sim_data_{ntrades}_gap_{min_gap}_seed_{seed}.pkl")
    # search for existing data file
    data_file = [file for file in os.listdir(datadir) if file.endswith(f"gap_{min_gap}_seed_{seed}.pkl")]
    if len(data_file) > 0:
        assert len(data_file) == 1, f"Multiple data files found for gap {min_gap} and seed {seed} in {datadir}"
        ntrades_existing = int(data_file[0].split('_')[2])
        assert ntrades_existing >= ntrades, f"Existing data has {ntrades_existing} trades, but need at least {ntrades} trades"
        mcmc_file = os.path.join(datadir, data_file[0])
        print(f"Loading existing simulated data from {mcmc_file}")
        with open(mcmc_file, 'rb') as f:
            S, v_sim = pickle.load(f)
    else:
        
        S, v_sim = simulate_data(params, sigma, tau, ntrades, data_method, min_gap, seed)
        mcmc_file = os.path.join(datadir, f"sim_data_{ntrades}_gap_{min_gap}_seed_{seed}.pkl")
        # Save simulated data
        with open(mcmc_file, 'wb') as f:
            pickle.dump((S, v_sim), f)

    traces = inference_tim_reparam_online(ntrades_list, S, v_sim, sigma, tau, min_gap, seed, outdir, prior_params, **mcmc_kwargs)

    return traces

def process_simulation_online_starmap(ntrades_list, params, sigma, tau, data_method, min_gap, seed, outdir, datadir, prior_params, mcmc_kwargs):
    return process_simulation_online(ntrades_list, params, sigma, tau, data_method, min_gap, seed, outdir, datadir, prior_params, **mcmc_kwargs)

def log_mcmc_info(args, output_folder, outdir, datadir, ntrades_list, num_runs, seed, mcmc_kwargs, prior_params):
    info_data = {datetime.now().strftime('%Y%m%d_%H%M%S'):
        {
        'input_parameters': {
            'output_folder': output_folder,
            'trade_list': ntrades_list,
            'seed': seed,
            'num_runs': num_runs,
            'kappa': args.kappa,
            'rho': args.rho,
            'gamma': args.gamma,
            'sigma': args.sigma,
            'tau': args.tau,
            'data_method': args.data_method,
            'min_gap': args.min_gap,
            'debug': args.debug
        },
        'mcmc_parameters': mcmc_kwargs,
        'prior_parameters': prior_params,
        'output_directories': {
            'outdir': outdir,
            'datadir': datadir
        }
        
    }
        }
    
    # Save info file - update existing dictionary or create new
    info_file = os.path.join(outdir, f'run_info.json')
    
    if os.path.exists(info_file):
        # Load existing data
        with open(info_file, 'r') as f:
            existing_data = json.load(f)
        
        # Update existing dictionary with new run data
        existing_data.update(info_data)
        
        # Save updated data
        with open(info_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        print(f"Run info updated in existing file: {info_file}")
    else:
        # Create new file with current run data
        with open(info_file, 'w') as f:
            json.dump(info_data, f, indent=2)
        print(f"New run info file created: {info_file}")
    
    return info_file

# ─── Main Execution ───
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TIM Bayesian inference with MCMC')
    parser.add_argument('--output_folder', type=str, 
                       default='20251008_debug', 
                       help='Output folder name for results (default: TIMESTAMP)') # f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser.add_argument('--ntrades_list', type=int, nargs='+', default=[25],
                       help='List of trade sizes to simulate (default: [100])')
    parser.add_argument('--data_method', type=str, default='gaps',
                       help='Method for simulating trade times: gaps, poisson, gaussian, poisson_gaussian (default: gaps)')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of simulation runs (default: 15)')
    parser.add_argument('--kappa', type=float, default=5e-5,
                       help='Transient impact scale (default: 1e-5)')
    parser.add_argument('--rho', type=float, default=2.231,
                       help='Decay rate (default: 2.231)')
    parser.add_argument('--gamma', type=float, default=5e-5,
                       help='Permanent impact scale (default: 1e-5)')
    parser.add_argument('--sigma', type=float, default=0.95,
                       help='Volatility (default: 0.45)')
    parser.add_argument('--tau', type=float, default=0.01,
                       help='Time step (default: 0.01)')
    parser.add_argument('--min_gap', type=int, default=100,
                       help='Minimum gap between trades (default: 100)')
    parser.add_argument('--draws', type=int, default=1000,
                          help='Number of MCMC draws')
    parser.add_argument('--tune', type=int, default=4000,
                        help='Number of MCMC tuning steps')
    parser.add_argument('--cores', type=int, default=8,
                        help='Number of CPU cores for MCMC')
    parser.add_argument('--chains', type=int, default=8,
                        help='Number of MCMC chains')
    parser.add_argument('--seed', type=int, default=100,
                       help='Random seed (default: 100)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for additional output (default: False)')
    
    # Prior parameters
    parser.add_argument('--t_mean_prior', type=float, default=2e-5,
                       help='Mean parameter for theta log-normal prior (default: 2e-5)')
    parser.add_argument('--t_std_prior', type=float, default=1,
                       help='Standard deviation parameter for theta log-normal prior (default: 1)')
    parser.add_argument('--alpha_a_prior', type=float, default=2,
                       help='Alpha parameter for alpha beta prior (default: 2)')
    parser.add_argument('--alpha_b_prior', type=float, default=2,
                       help='Beta parameter for alpha beta prior (default: 1)')
    parser.add_argument('--rho_mean_prior', type=float, default=2.231,
                       help='Mean parameter for rho log-normal prior (default: 8)')
    parser.add_argument('--rho_std_prior', type=float, default=20,
                       help='Standard deviation parameter for rho log-normal prior (default: 2)')
    
    args = parser.parse_args()
    
    # Start timer
    start_time = time.time()
    impact_model = 'TIM'
    
    # Create output directories
    source_path = '/nfs/home/dannis/AC-model/TIM'

    output_folder = args.output_folder if not args.debug else f"debug"
    
    outdir = os.path.join(source_path, f"{impact_model}_MCMC_Results", output_folder)
    datadir = os.path.join(source_path, f"{impact_model}_simulated_data", output_folder)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    # Parameters from arguments
    ntrades_list = args.ntrades_list
    num_runs = args.num_runs
    params = {
        "kappa": args.kappa,
        "rho": args.rho,
        "gamma": args.gamma
    }
    sigma = args.sigma
    tau = args.tau
    data_method = args.data_method
    min_gap = args.min_gap
    seed = args.seed
    
    # prior parameters
    t_mean_prior = args.t_mean_prior
    t_std_prior = args.t_std_prior
    alpha_a_prior = args.alpha_a_prior
    alpha_b_prior = args.alpha_b_prior
    rho_mean_prior = args.rho_mean_prior
    rho_std_prior = args.rho_std_prior
    prior_params = {
        "t_mean_prior": t_mean_prior,
        "t_std_prior": t_std_prior,
        "alpha_a_prior": alpha_a_prior,
        "alpha_b_prior": alpha_b_prior,
        "rho_mean_prior": rho_mean_prior,
        "rho_std_prior": rho_std_prior,}
    # MCMC parameters
    draws = args.draws
    tune = args.tune
    cores = args.cores
    chains = args.chains
    
    mcmc_kwargs = {
        "draws": draws,
        "tune": tune,
        "cores": cores if num_runs == 1 else 1,  # single core per process if parallelizing runs
        "chains": chains,
        "target_accept": 0.99,
        "return_inferencedata": True
    }  

    # Create info dictionary with all parameters and timing
    info_file = log_mcmc_info(args, output_folder, outdir, datadir, ntrades_list, num_runs, seed, mcmc_kwargs, prior_params)
    
    
    args_list = [
        (ntrades_list, params, sigma, tau, data_method, min_gap, seed + run, outdir, datadir, prior_params,mcmc_kwargs)
        for run in range(num_runs)
    ]

    if len(args_list) == 1:
        process_simulation_online_starmap(*args_list[0])
    # Parallelize the double loop
    else:
        with Pool() as pool:
            pool.starmap(process_simulation_online_starmap, args_list)

   # End timer
    end_time = time.time()
    run_time = end_time - start_time
    # Load existing data
    with open(info_file, 'r') as f:
        existing_data = json.load(f)
    
    # Update existing dictionary with new run data
    existing_data['total_runtime_seconds'] = run_time

    # Save updated data
    with open(info_file, 'w') as f:
        json.dump(existing_data, f, indent=2)

    
    print(f"Elapsed time: {run_time:.6f} seconds")


    
# Run the script
if __name__ == "__main__":
    main()