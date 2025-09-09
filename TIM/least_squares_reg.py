import numpy as np
from scipy.optimize import least_squares
import arviz as az
import pickle
from tqdm import tqdm
import pandas as pd
import pickle
import argparse
from data_syn import fn_dI, fn_dS
import os
from multiprocessing import Pool

def build_I(n, kappa, rho, tau):
    I = np.zeros(len(n) + 1)  # I_0 aligns with index 0 (unused)
    a = 1.0 - rho*tau
    for k in range(1, len(n)+1):
        I[k] = a*I[k-1] + kappa*n[k-1]*tau    # produces I_k for k >= 1
    return I

def fn_dS0(theta, n, tau):
    gamma, kappa, rho = theta                 
    I = build_I(n, kappa, rho, tau)    # I_k, k=0..K (I_0=0)
    # r_k = ΔS_k + (γ+κ) n_k - ρ τ I_{k-1}
    dS0 = -((gamma + kappa) * n * tau - (rho * tau) * I[:-1])
    return dS0, I



def residuals(theta, n, S, tau):
    dS = np.diff(S)                    # ΔS_k, k=1..K
    dS0 = fn_dS0(theta, n, tau)[0]
    r = dS - dS0
    return r

def fit(n, S, tau, theta0=(1e-3, 1e-3, 1.0), bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, 1.0/0.01 - 1e-8])):
    res = least_squares(residuals, x0=np.array(theta0, float),
                        args=(np.asarray(n, float), np.asarray(S, float), float(tau)),
                        bounds=bounds, method="trf", jac="2-point")
    theta_hat = res.x
    # variance & standard errors (Gaussian NLS)
    r = res.fun
    dof = max(1, len(r) - len(theta_hat))
    sigma2 = (r @ r) / dof
    JTJ_inv = np.linalg.pinv(res.jac.T @ res.jac)
    cov = sigma2 * JTJ_inv
    se = np.sqrt(np.diag(cov))
    return {
        "theta_hat": {"gamma": theta_hat[0], "kappa": theta_hat[1], "rho": theta_hat[2]},
        "se": {"gamma": se[0], "kappa": se[1], "rho": se[2]},
        "sigma2_hat": sigma2,
        "success": res.success,
        "message": res.message
    }

def run_fit(data_file):
    n_trades = int(data_file.split('_')[2])
    seed = int(data_file.split('_')[-1].split('.')[0])
    tau = 0.01
    print(f"Processing file: {data_file}")
    with open(os.path.join(args.data_dir, data_file), "rb") as f:
        data = pickle.load(f)
        S = data[0]
        n = data[1]
    result = fit(n, S, tau)
    with open(f'{args.data_dir}/least_squares_results/fit_result_{n_trades}_seed_{seed}.pkl', 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit model to trading data")
    parser.add_argument("--data_dir", type=str, 
                        default='/nfs/home/dannis/AC-model/TIM/TIM_simulated_data/20250826',
                        help="Path to the folder containing trading data file")
    parser.add_argument('--trade_list', type=int, nargs='+', default=[50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
                       help='List of trade sizes to simulate (default: [100])')
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of simulation runs (default: 10)')
    args = parser.parse_args()
    
    data_files = []
    for f in os.listdir(args.data_dir):
        for n_trade in args.trade_list:
            for seed in range(100, 100 + args.num_runs):
                if f == f'sim_data_{n_trade}_gap_100_seed_{seed}.pkl':
                    data_files.append(f)

    os.makedirs(f'{args.data_dir}/least_squares_results', exist_ok=True)
    # Parallelize the double loop
    with Pool() as pool:
        pool.map(run_fit, data_files)
    
    results_all = []
    for result_file in os.listdir(f'{args.data_dir}/least_squares_results'):
        if result_file.startswith('fit_result_') and result_file.endswith('.pkl'):
            with open(os.path.join(f'{args.data_dir}/least_squares_results', result_file), 'rb') as f:
                res = pickle.load(f)
                n_trades = int(result_file.split('_')[2])
                seed = int(result_file.split('_')[-1].split('.')[0])
                res_flat = {
                    'n_trades': n_trades,
                    'seed': seed,
                    'gamma_hat': res['theta_hat']['gamma'],
                    'kappa_hat': res['theta_hat']['kappa'],
                    'rho_hat': res['theta_hat']['rho'],
                    'gamma_se': res['se']['gamma'],
                    'kappa_se': res['se']['kappa'],
                    'rho_se': res['se']['rho'],
                    'sigma2_hat': res['sigma2_hat'],
                    'success': res['success'],
                    'message': res['message']
                }
                results_all.append(res_flat)
                
    df_results = pd.DataFrame(results_all)
    df_results.to_csv(f'{args.data_dir}/least_squares_results/summary.csv', index=False)

