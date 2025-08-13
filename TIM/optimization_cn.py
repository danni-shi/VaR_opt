import math
import numpy as np
import os
import arviz as az
import gurobipy as gp
from gurobipy import GRB
import pickle
import datetime
import json
import argparse

## Posterior Sampling
def sample_posteriors(trace_file, n_scenarios):
    print(f"\nProcessing trace file: {trace_file}")
    idata = az.from_netcdf(trace_file)
    kappa_vals = idata.posterior['kappa'].values.reshape(-1)
    gamma_vals = idata.posterior['gamma'].values.reshape(-1)
    rho_vals = idata.posterior['rho'].values.reshape(-1)
    n_samples = kappa_vals.size
    idx = np.random.randint(0, n_samples, size=n_scenarios)
    posterior_samples = {
        "kappa": kappa_vals[idx] * 100,
        "gamma": gamma_vals[idx] * 100,
        "rho": rho_vals[idx]
    }
    noise = np.random.normal(0, 1, (n_scenarios, n_steps + 1))
    print(f"Mean of Kappa: {np.mean(posterior_samples['kappa'])}, STD: {np.std(posterior_samples['kappa'])}")
    print(f"Mean of Gamma: {np.mean(posterior_samples['gamma'])}, STD: {np.std(posterior_samples['gamma'])}")
    print(f"Mean of rho: {np.mean(posterior_samples['rho'])}, STD: {np.std(posterior_samples['rho'])}")
    return posterior_samples, noise

## Proxy Normal Distribution Sampling
def sample_proxy(n_scenarios, theta_m, theta_v , alpha_a, alpha_b, rho_m=2.231, rho_v=0):
        if theta_v == 0.:
            theta_samples = np.array([theta_m] * n_scenarios)
        else:
            theta_samples = np.random.normal(theta_m, theta_v, n_scenarios)
        if alpha_a == 0. and alpha_b == 0.:
            alpha_samples = np.array([0.5] * n_scenarios)
        else:
            alpha_samples = np.random.beta(alpha_a, alpha_b, n_scenarios) # Sample from Beta distribution or Normal Distribution?
        rho_samples = np.random.normal(rho_m, rho_v, n_scenarios)
        # Transformations to get model parameters
        posterior_samples = {
            "kappa": theta_samples * alpha_samples * 100,     # scale as before
            "gamma": theta_samples * (1 - alpha_samples) * 100,
            "rho": rho_samples
        }
        noise = np.random.normal(0, 1, (n_scenarios, n_steps + 1))
        print(f"Mean of Kappa: {np.mean(posterior_samples['kappa'])}, STD: {np.std(posterior_samples['kappa'])}")
        print(f"Mean of Gamma: {np.mean(posterior_samples['gamma'])}, STD: {np.std(posterior_samples['gamma'])}")
        print(f"Mean of rho: {np.mean(posterior_samples['rho'])}, STD: {np.std(posterior_samples['rho'])}")
        return posterior_samples, noise

def IS_tim_particles(action, posterior_samples, noise, n_scenarios, n_data, total_qty, tau, eta, sigma, spread):
    IS_exprs = []
    for p in range(n_scenarios):
        kappa = posterior_samples["kappa"][p]
        gamma = posterior_samples["gamma"][p]
        rho = 2.231  # Or posterior_samples["rho"][p]
        permanent_impact = 0.5 * gamma * total_qty**2 - 0.5 * gamma * gp.quicksum(action[k] * action[k] for k in range(n_data + 1))
        spread_cost = spread * total_qty
        temporary_impact = (eta / tau) * gp.quicksum(action[k] * action[k] for k in range(n_data + 1))
        stochastic_term = 0.0
        for k in range(n_data + 1):
            inventory_expr_k = total_qty - gp.quicksum(action[i] for i in range(k + 1))
            stochastic_term += inventory_expr_k * noise[p, k]
        stochastic_term = sigma * math.sqrt(tau) * stochastic_term
        transient_impact = kappa * gp.quicksum(
            action[k] * (gp.quicksum(action[i] * math.exp(-rho * tau * (k - i)) for i in range(k)))
            for k in range(n_data + 1)
        )
        IS_p = permanent_impact + spread_cost + temporary_impact - stochastic_term + transient_impact
        IS_exprs.append(IS_p)
    return IS_exprs

def run_optimization(posterior_samples, noise, n_scenarios, n_data, total_qty, tau, eta, sigma, spread, chance_thresh, VaR_thresh, M=5e3):
    model = gp.Model("TIM_MCMC_Opt")
    model.Params.OutputFlag = 1
    model.Params.NonConvex = 2
    model.Params.Cuts = 2
    model.Params.Presolve = 2
    model.Params.MIPFocus = 2
    model.Params.OBBT = 2
    model.Params.Threads = 90
    model.Params.MIPGap = 0.002
    model.Params.TimeLimit = 10800

    action = model.addVars(n_data + 1, lb=0, name="action")
    b = model.addVars(n_scenarios, vtype=GRB.BINARY, name="b")
    model.addConstr(gp.quicksum(action[k] for k in range(n_data + 1)) == total_qty, "TotalShares")
    IS_exprs = IS_tim_particles(action, posterior_samples, noise, n_scenarios, n_data, total_qty, tau, eta, sigma, spread)
    for p, IS_p in enumerate(IS_exprs):
        model.addQConstr(IS_p <= VaR_thresh + M * b[p], name=f"IS_{p}")
    model.addConstr(gp.quicksum(b[p] for p in range(n_scenarios)) <= chance_thresh * n_scenarios, "Chance")
    model.setObjective((1.0 / n_scenarios) * gp.quicksum(IS_exprs), GRB.MINIMIZE)
    model.optimize()

    # Prepare per-run results
    inventory_plots = []
    errors = {}
    kappa_samples = posterior_samples["kappa"]
    gamma_samples = posterior_samples["gamma"]
    rho_samples = posterior_samples["rho"]

    if model.status in (2, 9, 13):
        trades = [action[k].X for k in range(n_data + 1)]
        b_arr = np.array([b[p].X for p in range(n_scenarios)])
        obj_val = model.ObjVal
        mip_gap = model.MIPGap
        runtime = model.Runtime
        tail_prob = b_arr.mean()
    else:
        trades = [np.nan] * (n_data + 1)
        errors["status"] = model.status
        obj_val = np.nan
        mip_gap = np.nan
        runtime = np.nan
        tail_prob = np.nan

    # Inventory trajectory
    inv = total_qty
    inv_traj = [total_qty]
    if model.status in (2, 9, 13):
        for q in trades:
            inv -= q
            inv_traj.append(inv)
    inventory_plots.append(inv_traj)

    results = {
        "status": model.status,
        "trades": trades,
        "obj_val": obj_val,
        "mip_gap": mip_gap,
        "runtime": runtime,
        "tail_prob": tail_prob,
        "inventory": inventory_plots,
        "errors": errors,
        "kappa_samp": kappa_samples,
        "gamma_samp": gamma_samples,
        "rho_samp": rho_samples
    }
    return results

# -------------------------
# Example Usage in Main Loop
# -------------------------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TIM optimization with Proxy Distribution')
    parser.add_argument('--outdir', type=str, 
                       default=f"TIM_fix_mean_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                       help='Output directory for results (default: TIM_opt_results_TIMESTAMP)')
    
    parser.add_argument('--var_list', type=float, nargs='+', default=[1.41e-06, 27.93],
                       help='Variance for the corresponding proxy distribution')
    parser.add_argument('--var_idx', type=int, default=1, help='index of variance parameters')
    
    parser.add_argument('--run_seeds', type=int, nargs='+', 
                       default=[115,116,117], 
                       help='List of seeds for random number generation (default: [100, 101, 102])')
    parser.add_argument('--n_scenarios', type=int, default=50)
    
    args = parser.parse_args()
    
    # Use the command line argument for output directory
    outdir = args.outdir
    # var_list = args.var_list
    if args.var_idx == 0:
        var_list = [1.349766e-06, 4.397102] # n_trades = 250
    elif args.var_idx == 1:
        var_list = [7.091207e-07, 5.213667] # n_trades = 800
    elif args.var_idx == 2:
        var_list = [5.953597e-07, 5.379057] # n_trades = 1100
    elif args.var_idx == 3:
        var_list = [5.268546e-07, 6.088576] # n_trades = 1400
    else: var_list = [0., 0.]

    run_seeds = args.run_seeds
    #mcmc_timestamp = args.mcmc_timestamp
    n_scenarios = args.n_scenarios

    os.makedirs(outdir, exist_ok=True)
    #mcmc_results_dir = "/nfs/home/dannis/AC-model/TIM/TIM_MCMC_Results"  # Example directory, adjust as needed
    # mcmc_timestamp = "20250620_162746"
    # trace_files = os.listdir(f'{mcmc_results_dir}/{mcmc_timestamp}')[-2:]
    n_steps = 10
    # n_scenarios = 500
    
    total_qty = 1e3
    T = 1
    tau = T / n_steps
    spread = 0
    eta = 1e-03 * tau
    sigma = 0.45
    chance_thresh = 0.1
    M = 5e3
    VaR_thresh = 1001
    # num_runs = 2
    # run_seeds = [100 + run for run in range(num_runs)]
    # trade_list = [500, 800]
    gap = 100 #??


    theta_v, alpha_ab = var_list[0], var_list[1]
    opt_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(outdir, f"{args.var_idx}_theta_{theta_v}_alpha_{alpha_ab}")
    os.makedirs(run_folder, exist_ok=True)
    config = {'opt_timestamp': opt_timestamp, 
                'theta_var': theta_v, 
                'alpha_var': alpha_ab,
                'n_scenarios': n_scenarios, 
                'gap': gap}
    
    with open(os.path.join(run_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    for seed in run_seeds:

        np.random.seed(seed)
        # trace_file = f'trace_{n_trade}_gap_{gap}_seed_{seed}.nc'
        # print(f"\nProcessing trace file: {trace_file}")
        # posterior_samples, noise = sample_posteriors(f'{mcmc_results_dir}/{mcmc_timestamp}/{trace_file}', n_scenarios)
        posterior_samples, noise = sample_proxy(n_scenarios, theta_m= 2e-05, theta_v =theta_v, alpha_a=alpha_ab, alpha_b = alpha_ab, rho_m=2.231, rho_v=0)
        

        print(f"Seed Number: {seed}")
        results = run_optimization(
            posterior_samples, noise, n_scenarios, n_steps, total_qty, tau, eta, sigma, spread, chance_thresh, VaR_thresh
        )
        print(f"Optimization Status: {results['status']}")
        # Save results for this run
        results_file = os.path.join(run_folder, f"opt_{theta_v}_{seed}.pkl")
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
