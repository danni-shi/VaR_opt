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

def sample_posteriors(trace_file, n_scenarios, n_steps=10):
    print(f"\nProcessing trace file: {trace_file}")
    idata = az.from_netcdf(trace_file)
    kappa_vals = idata.posterior['kappa'].values.reshape(-1)
    gamma_vals = idata.posterior['gamma'].values.reshape(-1)
    rho_vals = idata.posterior['rho'].values.reshape(-1)
    n_samples = kappa_vals.size
    idx = np.random.randint(0, n_samples, size=n_scenarios)
    posterior_samples = {
        "kappa": kappa_vals[idx],
        "gamma": gamma_vals[idx],
        "rho": rho_vals[idx]
    }
    noise = np.random.normal(0, 1, (n_scenarios, n_steps + 1))
    print(f"Mean of Kappa: {np.mean(posterior_samples['kappa'])}, STD: {np.std(posterior_samples['kappa'])}")
    print(f"Mean of Gamma: {np.mean(posterior_samples['gamma'])}, STD: {np.std(posterior_samples['gamma'])}")
    print(f"Mean of rho: {np.mean(posterior_samples['rho'])}, STD: {np.std(posterior_samples['rho'])}")
    return posterior_samples, noise

def IS_tim_particles(action, posterior_samples, noise, n_scenarios, n_data, total_qty, tau, q, sigma, spread):
    IS_exprs = []
    for p in range(n_scenarios):
        kappa = posterior_samples["kappa"][p]
        gamma = posterior_samples["gamma"][p]
        rho = 2.231  # Or posterior_samples["rho"][p]
        permanent_impact = 0.5 * gamma * total_qty**2 - 0.5 * gamma * gp.quicksum(action[k] * action[k] for k in range(n_data + 1))
        spread_cost = spread * total_qty
        temporary_impact = (1/2 * (gamma + kappa)) * gp.quicksum(action[k] * action[k] for k in range(n_data + 1)) # should be estimated
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

def run_optimization(posterior_samples, noise, n_scenarios, n_steps, total_qty, tau, eta, sigma, spread, chance_thresh, VaR_thresh, M=5e3):
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

    action = model.addVars(n_steps + 1, lb=0, name="action")
    b = model.addVars(n_scenarios, vtype=GRB.BINARY, name="b")
    model.addConstr(gp.quicksum(action[k] for k in range(n_steps + 1)) == total_qty, "TotalShares")
    IS_exprs = IS_tim_particles(action, posterior_samples, noise, n_scenarios, n_steps, total_qty, tau, eta, sigma, spread)
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
        trades = [action[k].X for k in range(n_steps + 1)]
        b_arr = np.array([b[p].X for p in range(n_scenarios)])
        obj_val = model.ObjVal
        mip_gap = model.MIPGap
        runtime = model.Runtime
        tail_prob = b_arr.mean()
    else:
        trades = [np.nan] * (n_steps + 1)
        errors["status"] = model.status
        mip_gap = np.nan
        obj_val = np.nan
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
    parser = argparse.ArgumentParser(description='Run TIM optimization with MCMC results')
    parser.add_argument('--outdir', type=str, 
                       default=f"TIM_opt_results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                       help='Output directory for results (default: TIM_opt_results_TIMESTAMP)')
    parser.add_argument('--trade_list', type=int, nargs='+', default=[2000],
                       help='List of trade sizes to optimize (default: [2000])')
    parser.add_argument('--run_seeds', type=int, nargs='+', 
                       default=[115], 
                       help='List of seeds for random number generation (default: [100, 101, 102])')
    parser.add_argument('--mcmc_timestamp', type=str, default="20250701_134415",
                        help='Timestamp of the MCMC results directory')
    parser.add_argument('--n_scenarios', type=int, default=100)
    parser.add_argument('--loss_prob', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.)
    parser.add_argument('--loss_thresh', type=float, default=9e5)
    parser.add_argument('--M', type=float, default=1e6)
    
    args = parser.parse_args()
    
    # Use the command line argument for output directory
    outdir = args.outdir
    trade_list = args.trade_list
    run_seeds = args.run_seeds
    mcmc_timestamp = args.mcmc_timestamp
    n_scenarios = args.n_scenarios
    loss_thresh = args.loss_thresh
    chance_thresh = args.loss_prob - args.margin
    M = args.M

    os.makedirs(outdir, exist_ok=True)
    mcmc_results_dir = "/nfs/home/dannis/AC-model/TIM/TIM_MCMC_Results"  # Example directory, adjust as needed
    # trace_files = os.listdir(f'{mcmc_results_dir}/{mcmc_timestamp}')[-2:]
    n_steps = 10
    
    total_qty = 1e5
    T = 1
    tau = T / n_steps
    spread = 0
    q = 5000
    sigma = 0.95
    # loss_thresh = 1 / (2 * q) * (total_qty**2 / T) * 1.001  # Slightly above the deterministic cost of liquidation at the first
    
    # num_runs = 2
    # run_seeds = [100 + run for run in range(num_runs)]
    # trade_list = [500, 800]
    gap = 100
    for n_trade in trade_list:
        opt_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(outdir, f"ntrades{n_trade}_nscenarios{n_scenarios}_thresh{int(chance_thresh*100)}_M{int(M)}")
        # run_folder = os.path.join(outdir, f"{n_trade}_nscenarios_{n_scenarios}")
        os.makedirs(run_folder, exist_ok=True)
        config = {
            'opt_timestamp': opt_timestamp, 
            'mcmc_timestamp': mcmc_timestamp,
            'n_trade': n_trade, 
            'n_scenarios': n_scenarios, 
            'gap': gap,
            'n_steps': n_steps,
            'total_qty': total_qty,
            'tau': tau,
            'spread': spread,
            'q': q,
            'sigma': sigma,
            'chance_thresh': chance_thresh,
            'loss_thresh': loss_thresh,
            'M': M
        }

        with open(os.path.join(run_folder, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
        for seed in run_seeds:
    
            np.random.seed(seed)
            trace_file = f'trace_{n_trade}_gap_{gap}_seed_{seed}.nc'
            print(f"\nProcessing trace file: {trace_file}")
            posterior_samples, noise = sample_posteriors(f'{mcmc_results_dir}/{mcmc_timestamp}/{trace_file}', n_scenarios, n_steps)
            
            print(f"Seed Number: {seed}")
            results = run_optimization(
                posterior_samples, noise, n_scenarios, n_steps, total_qty, tau, q, sigma, spread, chance_thresh, loss_thresh
            )
            print(f"Optimization Status: {results['status']}")
            # Save results for this run
            results_file = os.path.join(run_folder, f"opt_{trace_file[:-3]}.pkl")
            with open(results_file, "wb") as f:
                pickle.dump(results, f)