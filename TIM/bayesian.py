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

def inference_tim_reparam(S, v_sim, tau, sigma_true):
    """
    Runs the Bayesian inference model using PyMC with reparameterization.
    """
    # Prepare data
    v_data = v_sim         # shape (T,)
    S_data = S             # shape (T+1,)
    dS_data = S_data[1:] - S_data[:-1]  # increments shape (T,)

    # Define priors
    t_mu, t_sigma = lognormal_params(5e-4, 2e-4)  # for θ
    a_alpha, b_alpha = 2, 1                        # for α∈[0,1]
    r_mu, r_sigma = lognormal_params(8, 2)        # for rho

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
            tune=3000,
            cores = 1,
            chains = 8,
            target_accept=0.99, 
            return_inferencedata=True
        )
        return trace
    
# ─── Worker Function for Parallel Execution ───
def process_simulation(n_trades, params, sigma, tau, min_gap, seed, outdir, datadir):
    """
    Worker function to simulate data and perform Bayesian inference for a single combination of n_trades and run.
    """
    #n_trades, run, params, sigma, tau, min_gap, seed, outdir, datadir = args
    S, v_sim = simulate_data(params, sigma, tau, n_trades, min_gap, seed)

    # Save simulated data to pickle file
    output_file = os.path.join(datadir, f"sim_data_{n_trades}_gap_{min_gap}_seed_{seed}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump((S, v_sim), f)

    # Perform Bayesian inference
    # trace = inference_tim(S, v_sim, tau, sigma, outdir, n_trades, min_gap, seed)
    trace = inference_tim_reparam(S, v_sim, tau, sigma)

    # Save results
    trace.to_netcdf(os.path.join(outdir, f"trace_{n_trades}_gap_{min_gap}_seed_{seed}.nc"))

# ─── Main Execution ───
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TIM Bayesian inference with MCMC')
    parser.add_argument('--output_folder', type=str, 
                       default=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                       help='Output folder name for results (default: TIMESTAMP)')
    parser.add_argument('--trade_list', type=int, nargs='+', default=[100],
                       help='List of trade sizes to simulate (default: [100])')
    parser.add_argument('--num_runs', type=int, default=15,
                       help='Number of simulation runs (default: 15)')
    parser.add_argument('--kappa', type=float, default=1e-5,
                       help='Transient impact scale (default: 1e-5)')
    parser.add_argument('--rho', type=float, default=2.231,
                       help='Decay rate (default: 2.231)')
    parser.add_argument('--gamma', type=float, default=1e-5,
                       help='Permanent impact scale (default: 1e-5)')
    parser.add_argument('--sigma', type=float, default=0.45,
                       help='Volatility (default: 0.45)')
    parser.add_argument('--tau', type=float, default=0.01,
                       help='Time step (default: 0.01)')
    parser.add_argument('--min_gap', type=int, default=100,
                       help='Minimum gap between trades (default: 100)')
    parser.add_argument('--seed', type=int, default=100,
                       help='Random seed (default: 100)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for additional output (default: False)')
    
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
    trade_list = args.trade_list
    num_runs = args.num_runs
    params = {
        "kappa": args.kappa,
        "rho": args.rho,
        "gamma": args.gamma
    }
    sigma = args.sigma
    tau = args.tau
    min_gap = args.min_gap
    seed = args.seed

    # Prepare arguments for parallel execution
    args_list = [
        (n_trades, params, sigma, tau, min_gap, seed+run, outdir, datadir)
        for n_trades in trade_list
        for run in range(num_runs)
    ]

    # Parallelize the double loop
    with Pool() as pool:
        pool.starmap(process_simulation, args_list)
    
    # simulation_times = []
    # start_time = time.time()
    # for n_trades in trade_list:
    #     for run in range(num_runs):
    #         start_time_run = time.time()
    #         function_args = (n_trades, params, sigma, tau, min_gap, seed+run, outdir, datadir)
    #         process_simulation(*function_args)
    #         end_time_run = time.time()
    #         sim_runtime = end_time_run - start_time_run
    #         sim_seed = seed + run
    #         simulation_times.append({
    #             'n_trades': n_trades,
    #             'seed': sim_seed,
    #             'runtime_seconds': sim_runtime
    #         })
            
 # End timer
    end_time = time.time()
    run_time = end_time - start_time
    
    # Create info dictionary with all parameters and timing
    info_data = {seed:
        {
        'run_info': {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_runtime_seconds': run_time,
            'total_simulations': len(args_list)
        },
        'input_parameters': {
            'output_folder': output_folder,
            'trade_list': trade_list,
            'num_runs': num_runs,
            'kappa': args.kappa,
            'rho': args.rho,
            'gamma': args.gamma,
            'sigma': sigma,
            'tau': tau,
            'min_gap': min_gap,
            'debug': args.debug
        },
        'output_directories': {
            'outdir': outdir,
            'datadir': datadir
        }}
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
    
    print(f"Elapsed time: {run_time:.6f} seconds")
    
# Run the script
if __name__ == "__main__":
    main()