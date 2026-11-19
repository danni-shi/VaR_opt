import math
from xml.parsers.expat import model
import numpy as np
import os
import arviz as az
import gurobipy as gp
from gurobipy import GRB
import pickle
import datetime
import json
import argparse
from itertools import product
from collections import defaultdict

from pydantic import TypeAdapter

MCMC_DIR = "/nfs/home/dannis/AC-model/TIM/TIM_MCMC_Results"  

def sample_posteriors(nTrades, mcmc_label, mcmc_timestamp, n_scenarios_to_add, scale, recenter, true_gamma, true_kappa, true_rho, gap=100):
    if mcmc_label == 'combined':
        trace_file = f'{MCMC_DIR}/{mcmc_timestamp}/nTrades{nTrades}_posteriors.pkl'
        
        if not os.path.isfile(trace_file):
            combined_posterior_samples = combine_posteriors(nTrades, gap, mcmc_timestamp)
            with open(trace_file, 'wb') as f:
                pickle.dump(combined_posterior_samples, f)

    else:
        trace_file = f'{MCMC_DIR}/{mcmc_timestamp}/trace_{nTrades}_gap_{gap}_seed_{mcmc_label}.nc'
        
    posterior_samples = sample_posteriors_from_single_file(trace_file, n_scenarios_to_add, scale)
    
    if recenter:
        # info_path = os.path.join(MCMC_DIR, mcmc_timestamp, 'run_info.json')
        # if os.path.isfile(info_path):
        #     with open(info_path, 'r') as f:
        #         run_info = json.load(f)
        #     true_params = {param: run_info["100"]["input_parameters"][param] for param in posterior_samples.keys()}
        # else:
        true_params = {'gamma': true_gamma, "kappa": true_kappa, "rho": true_rho}  # Default values if run_info.json is not available

        for param in posterior_samples.keys():
            posterior_samples[param] -= np.mean(posterior_samples[param])
            if param in ["kappa", "gamma"]:
                posterior_samples[param] += true_params[param] * scale
                assert np.isclose(np.mean(posterior_samples[param]), true_params[param] * scale, atol=1e-2), f"Recentering failed for {param}"
            else:
                posterior_samples[param] += true_params[param]
                assert np.isclose(np.mean(posterior_samples[param]), true_params[param], atol=1e-2), f"Recentering failed for {param}"
            
    
    return posterior_samples

def combine_posteriors(nTrades, gap, mcmc_timestamp, tim_params=['kappa', 'gamma', 'rho']):
    all_vals_dict = defaultdict(list)
    list_trace_files = [os.path.join(f'{MCMC_DIR}/{mcmc_timestamp}', file) \
                        for file in os.listdir(f'{MCMC_DIR}/{mcmc_timestamp}') \
                            if (file.startswith(f'trace_{nTrades}_gap_{gap}_seed_') or file.startswith(f'ndays_{nTrades}_seed_')) and file.endswith('.nc')]    
    for file in list_trace_files:  
        idata = az.from_netcdf(file)
        
        for param_name in tim_params:
            vals = idata.posterior[param_name].values.reshape(-1)
            all_vals_dict[param_name].append(vals)
    for param_name, vals in all_vals_dict.items():
        all_vals_dict[param_name] = np.concatenate(vals)
    return all_vals_dict

def sample_posteriors_from_single_file(trace_file, n_scenarios, scale):
    print(f"\nProcessing trace file: {trace_file}")
    
    if trace_file.endswith('.pkl'):
        with open(trace_file, 'rb') as f:
            all_vals_dict = pickle.load(f)
        kappa_vals = all_vals_dict['kappa'] * scale
        gamma_vals = all_vals_dict['gamma'] * scale
        rho_vals = all_vals_dict['rho']
    
    elif trace_file.endswith('.nc'):
        idata = az.from_netcdf(trace_file)
        kappa_vals = idata.posterior['kappa'].values.reshape(-1) * scale
        gamma_vals = idata.posterior['gamma'].values.reshape(-1) * scale
        rho_vals = idata.posterior['rho'].values.reshape(-1)
    
    else:
        raise NotImplementedError("Unsupported file format. Use .nc or .pkl files.")
    
    n_samples = kappa_vals.size
    idx = np.random.randint(0, n_samples, size=n_scenarios)
    posterior_samples = {
        "kappa": kappa_vals[idx],
        "gamma": gamma_vals[idx],
        "rho": rho_vals[idx]
    }
    
    print(f"Mean of Kappa: {np.mean(posterior_samples['kappa']):.3g}, STD: {np.std(posterior_samples['kappa']):.3g}")
    print(f"Mean of Gamma: {np.mean(posterior_samples['gamma']):.3g}, STD: {np.std(posterior_samples['gamma']):.3g}")
    print(f"Mean of rho: {np.mean(posterior_samples['rho']):.3g}, STD: {np.std(posterior_samples['rho']):.3g}")
    
    return posterior_samples

# def sample_posteriors_accumated(list_trace_files, n_scenarios, scale):
    
#     posterior_samples = {}

#     for i, param_name in enumerate(['kappa', 'gamma', 'rho']):
#         all_vals = []
#         for file in list_trace_files:  
#             idata = az.from_netcdf(file)
#             vals = idata.posterior[param_name].values.reshape(-1)
#             all_vals.append(vals)
#         all_vals = np.concatenate(all_vals) * scale
#         if i == 0:
#             idx = np.random.randint(0, len(all_vals), size=n_scenarios)
#         posterior_samples[param_name] = all_vals[idx]

#     print(f"Mean of Kappa: {np.mean(posterior_samples['kappa']):.3g}, STD: {np.std(posterior_samples['kappa']):.3g}")
#     print(f"Mean of Gamma: {np.mean(posterior_samples['gamma']):.3g}, STD: {np.std(posterior_samples['gamma']):.3g}")
#     print(f"Mean of rho: {np.mean(posterior_samples['rho']):.3g}, STD: {np.std(posterior_samples['rho']):.3g}")
    
#     return posterior_samples

def IS_tim_particles(action, posterior_samples, n_steps, total_qty, tau, q, sigma, spread, sample_rho):
    
    n_scenarios = len(posterior_samples['gamma'])
    
    IS_exprs = []
    
    noise = np.random.normal(0, 1, (n_scenarios, n_steps + 1))
    
    for p in range(n_scenarios):
        kappa = posterior_samples["kappa"][p]
        gamma = posterior_samples["gamma"][p]
        if sample_rho:
            rho = posterior_samples["rho"][p]
        else:
            rho = 2.231  
        permanent_impact = 0.5 * gamma * total_qty**2 - 0.5 * gamma * gp.quicksum(action[k] * action[k] for k in range(n_steps + 1))
        spread_cost = spread * total_qty
        temporary_impact = (1/2 * (gamma + kappa)) * gp.quicksum(action[k] * action[k] for k in range(n_steps + 1)) # should be estimated
        stochastic_term = 0.0
        for k in range(n_steps + 1):
            inventory_expr_k = total_qty - gp.quicksum(action[i] for i in range(k + 1))
            stochastic_term += inventory_expr_k * noise[p, k]
        stochastic_term = sigma * math.sqrt(tau) * stochastic_term
        transient_impact = kappa * gp.quicksum(
            action[k] * (gp.quicksum(action[i] * math.exp(-rho * tau * (k - i)) for i in range(k)))
            for k in range(n_steps + 1)
        )
        IS_p = permanent_impact + spread_cost + temporary_impact - stochastic_term + transient_impact
        IS_exprs.append(IS_p)
    return IS_exprs

def run_optimization(posterior_samples, n_steps, total_qty, tau, q, sigma, spread, chance_thresh, VaR_thresh, M, scale, sample_rho=False):
    
    n_scenarios = len(posterior_samples['gamma'])
    
    model = gp.Model("TIM_MCMC_Opt")
    model.Params.OutputFlag = 1
    model.Params.NonConvex = 2
    model.Params.Cuts = 2
    model.Params.Presolve = 2
    model.Params.MIPFocus = 2
    model.Params.OBBT = 2
    model.Params.Threads = 80
    model.Params.MIPGap = 0.002
    model.Params.TimeLimit = 10800
    
    # model.Params.DualReductions = 0

    action = model.addVars(n_steps + 1, lb=0, name="action")
    b = model.addVars(n_scenarios, vtype=GRB.BINARY, name="b")
    model.addConstr(gp.quicksum(action[k] for k in range(n_steps + 1)) == total_qty, "TotalShares")
    IS_exprs = IS_tim_particles(action, posterior_samples, n_steps, total_qty, tau, q, sigma, spread, sample_rho)
    for p, IS_p in enumerate(IS_exprs):
        model.addQConstr(IS_p - (VaR_thresh + M * b[p]) <= 0, name=f"IS_{p}")
    model.addConstr(gp.quicksum(b[p] for p in range(n_scenarios)) <= chance_thresh * n_scenarios, "Chance")
    model.setObjective((1.0 / n_scenarios) * gp.quicksum(IS_exprs), GRB.MINIMIZE)
    model.optimize()
    print(f"Optimization Status: {model.status}")
#     for c in model.getConstrs():
# #         print(c.getAttr("ConstrName"): )
#         violation = abs(c.getAttr("Slack"))
#         if violation > 1e-6:
#             print(f"{c.ConstrName} violated by {violation}")
    # model.computeIIS()
    # model.write("model.ilp")

    # Prepare per-run results
    inventory_plots = []
    errors = {}
    try:
        slack = model.getConstrs()[1].getAttr("Slack")/(chance_thresh * n_scenarios)
        print(f"Chance constraint slack: {slack:.2g}")
    except Exception as e:
        slack = np.nan
        print(f"Error retrieving chance constraint slack: {e}")
    if model.status in (2, 9, 13):
        try:
            trades = [action[k].X for k in range(n_steps + 1)]
            b_arr = np.array([b[p].X for p in range(n_scenarios)])
            obj_val = model.ObjVal
            mip_gap = model.MIPGap
            runtime = model.Runtime
            tail_prob = b_arr.mean()
        
        except Exception as e:
            trades = [np.nan] * (n_steps + 1)
            errors["status"] = model.status
            mip_gap = np.nan
            obj_val = np.nan
            runtime = np.nan
            tail_prob = np.nan
            print(f"No feasible solution available to extract: {e}")
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
        "obj_val": obj_val * scale,
        "mip_gap": mip_gap,
        "runtime": runtime,
        "tail_prob": tail_prob,
        "inventory": inventory_plots,
        "errors": errors,
        "chance_slack": slack
    }
    return results

# -------------------------
# Example Usage in Main Loop
# -------------------------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TIM optimization with MCMC results')
    parser.add_argument('--outdir', type=str, 
                       default=f"/nfs/home/dannis/AC-model/TIM_opt_results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                       help='Output directory for results (default: TIM_opt_results_TIMESTAMP)')
    parser.add_argument('--list_n_trades', type=int, nargs='+', default=[99999],
                       help='List of trade sizes to optimize')
    parser.add_argument('--mcmc_labels', type=str, nargs='+', 
                       default=['combined'], 
                       help='List of seeds for random number generation')
    parser.add_argument('--run_seeds', type=int, nargs='+', default=[100],
                       help='List of seeds for posterior sampling and optimization runs')
    parser.add_argument('--mcmc_timestamp', type=str, default="20250916_online_5e-5",
                        help='Timestamp of the MCMC results directory')
    parser.add_argument('--true_gamma', type=float, default=5e-5, help='True gamma value used in data generation')
    parser.add_argument('--true_kappa', type=float, default=0., help='True kappa value used in data generation')
    parser.add_argument('--list_sigma', type=float, nargs='+', default=[0.95], help='Volatility parameter (default: 0.95)')
    parser.add_argument('--n_scenarios', type=int,  nargs='+', default=[300])
    parser.add_argument('--loss_prob', type=float, default=0.1)
    parser.add_argument('--margins', type=float, nargs='+', default=[0.05])
    parser.add_argument('--loss_threshs', type=float, nargs='+', default=[4.5e5])
    parser.add_argument('--Ms', type=float, nargs='+', default=[5e5])
    parser.add_argument('--scales', type=float, nargs='+', default=[1], help='Scaling factor for quantities (default: 1.0)')
    parser.add_argument('--add_samples', action='store_true', help='Whether to add posterior samples if the original optimization is infeasible')
    parser.add_argument('--max_n_scenarios', type=int, default=400, help='Maximum number of scenarios to try if optimization is infeasible')
    parser.add_argument('--recenter', type=lambda x: TypeAdapter(bool).validate_python(x), default=True, help='Whether to recenter the posterior samples around their mean')
    parser.add_argument('--sample_rho', type=lambda x: TypeAdapter(bool).validate_python(x), default=False, help='Whether to sample rho from the posterior')
    parser.add_argument('--use_info_file', type=lambda x: TypeAdapter(bool).validate_python(x), default=False, help='Whether to read true parameters from run_info.json')

    args = parser.parse_args()
    print(type(args.use_info_file))
    print(args.use_info_file)
 # For reproducibility
    # Use the command line argument for output directory
    outdir = args.outdir
    run_seeds = args.run_seeds
    list_n_trades = args.list_n_trades
    list_mcmc_label = args.mcmc_labels
    mcmc_timestamp = args.mcmc_timestamp

    list_n_scenarios = args.n_scenarios
    list_scale = args.scales
    list_loss_thresh = args.loss_threshs
    list_chance_thresh = [args.loss_prob - margin for margin in args.margins]
    list_M = args.Ms
    list_sigma = args.list_sigma

    os.makedirs(outdir, exist_ok=True)
    
    # trace_files = os.listdir(f'{mcmc_results_dir}/{mcmc_timestamp}')[-2:]
    n_steps = 10

    total_qty_orig = 1e5
    T = 1
    tau = T / n_steps
    spread = 0.01
    q_orig = 1e4 # not really using this atm, should satisfy kappa + gamma = 1/q
    # adjust the loss thresholds based on the spread cost
    list_loss_thresh = [lt + spread * total_qty_orig for lt in list_loss_thresh]
    
    # sigma = args.sigma
    # read mcmc info file
    info_path = os.path.join(MCMC_DIR, mcmc_timestamp, 'run_info.json')
    
    if os.path.isfile(info_path) and args.use_info_file:
        with open(info_path, 'r') as f:
            run_info = json.load(f)
        true_gamma = run_info["100"]["input_parameters"]["gamma"]
        true_kappa = run_info["100"]["input_parameters"]["kappa"]
        true_rho = run_info["100"]["input_parameters"]["rho"]
        # sigma = run_info["100"]["input_parameters"]["sigma"]
        
    else:
        true_gamma = args.true_gamma
        true_kappa = args.true_kappa
        true_rho = 2.231
        print(f"Warning: run_info.json not found in {info_path}. Using input values.")
    # loss_thresh = 1 / (2 * q) * (total_qty**2 / T) * 1.001  # Slightly above the deterministic cost of liquidation at the first
    
    # num_runs = 2
    # run_seeds = [100 + run for run in range(num_runs)]
    # list_ntrades = [500, 800]
    gap = 100
    
    tim_params = list(product(list_n_trades, list_n_scenarios, list_loss_thresh, list_chance_thresh, list_M, list_mcmc_label, list_scale, list_sigma))

    # Iterate over all parameter combinations
    for n_trades, n_scenarios, loss_thresh_orig, chance_thresh, M_orig, mcmc_label, scale, sigma in tim_params:
        # for numerical stability, scale down some parameters
        loss_thresh = loss_thresh_orig / scale
        M = M_orig / scale
        total_qty = total_qty_orig / scale
        q = q_orig / scale
        
        # optimization
        opt_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(outdir, f"nTrades{n_trades}_nScenarios{n_scenarios}_lossThresh{int(loss_thresh_orig)}_chanceThresh{int(chance_thresh*100)}_M{int(M_orig)}_mcmcLabel{mcmc_label}_scale{int(scale)}_sigma{sigma}_")
        
        os.makedirs(run_folder, exist_ok=True)
        config = {
            'opt_timestamp': opt_timestamp, 
            'mcmc_timestamp': mcmc_timestamp,
            'n_trades': n_trades, 
            'n_scenarios': n_scenarios,
            'add_samples': args.add_samples,
            'max_n_scenarios': args.max_n_scenarios,
            'gap': gap,
            'n_steps': n_steps,
            'total_qty': total_qty_orig,
            'tau': tau,
            'spread': spread,
            'q': q_orig,
            'sigma': sigma,
            'gamma': true_gamma,
            'kappa': true_kappa,
            'rho': true_rho,
            'chance_thresh': chance_thresh,
            'loss_thresh': loss_thresh_orig,
            'M': M_orig,
            'mcmc_label': mcmc_label,
            'scale': scale,
            'recenter': args.recenter,
            'sample_rho': args.sample_rho,
            'use_info_file': args.use_info_file
        }

        with open(os.path.join(run_folder, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # run multiple times for each configuration
        for run_seed in run_seeds:
            
            np.random.seed(run_seed)
            
            max_n_scenarios = args.max_n_scenarios if args.add_samples else n_scenarios
            current_samples =  {
                                "kappa":[],
                                "gamma": [],
                                "rho": []
                            }
            
            n_scenarios_to_add = n_scenarios
            n_scenarios_opt = n_scenarios
            print("Starting optimization... ")
            print(f"Sampling from MCMC posterior: {mcmc_label}, seed: {run_seed}")
            while n_scenarios_to_add > 0 and n_scenarios_opt <= max_n_scenarios:
                if n_trades == 99999:
                    new_samples = {
                        "kappa": np.array([true_kappa * scale] * n_scenarios_to_add),
                        "gamma": np.array([true_gamma * scale] * n_scenarios_to_add),
                        "rho": np.array([true_rho] * n_scenarios_to_add)
                    }
                else:
                    new_samples = sample_posteriors(n_trades, mcmc_label, mcmc_timestamp, n_scenarios_to_add, scale, args.recenter, true_gamma, true_gamma, true_rho, gap)

                current_samples = {key: np.concatenate([current_samples[key], 
                                                        new_samples[key]]) \
                                                            for key in current_samples}
                
                assert n_scenarios_opt == len(current_samples['gamma'])
                print(f"\nRunning optimization with {n_scenarios} scenarios...")
                
                results = run_optimization(
                    current_samples, n_steps, total_qty, tau, 
                    q, sigma, spread, chance_thresh, loss_thresh, M=M, scale=scale, sample_rho=args.sample_rho
                )
                
                if results["status"] in (3, 4): # infeasible or unbounded
                    print(f"Optimization infeasible with {n_scenarios_opt} scenarios. Doubling scenarios and retrying...\n")
                    n_scenarios_to_add = min(n_scenarios_opt, max_n_scenarios - n_scenarios_opt)
                    n_scenarios_opt = n_scenarios_opt + n_scenarios_to_add
                else:
                    break
                    
                    
            
            # Save results for this run
            # print(f"\nFinished Processing trace file: {trace_file} on run seed {run_seed}")
            print(f'End of Optimization. \n\n')
            
            results_file = os.path.join(run_folder, f"nScenarios{n_scenarios_opt}_{run_seed}.pkl")
            with open(results_file, "wb") as f:
                pickle.dump(results, f)