#!/bin/bash

# Example usage:
# bash run_tim_opt.sh 20250620_162746

source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Set optional arguments
OUTDIR="/nfs/home/dannis/AC-model/TIM_opt_results/20251017_open_and_close"  # Output directory for results
MCMC_TIMESTAMP="20251014_open_and_close"
TRUE_GAMMA=5e-5
TRUE_KAPPA=5e-5
LIST_SIGMA=0.95  
MCMC_LABELS="combined"
RUN_SEEDS=$(echo {100..114})
N_SCENARIOS="1000"               # Example number of scenarios
LOSS_PROB=0.1                  # Example loss probability
LOSS_THRESH="4.5e5"  # Example loss thresholds
M="1e6"          # Example M values
SCALE="100"                     # Example scale factor
mkdir -p $OUTDIR


LIST_N_TRADES="80" 
MARGIN="0.06 0.07 0.08"    
# Call the Python script with arguments
python3 TIM/optimization.py \
    --outdir $OUTDIR \
    --mcmc_timestamp $MCMC_TIMESTAMP \
    --list_n_trades $LIST_N_TRADES \
    --mcmc_labels $MCMC_LABELS \
    --true_gamma $TRUE_GAMMA \
    --true_kappa $TRUE_KAPPA \
    --list_sigma $LIST_SIGMA \
    --run_seeds $RUN_SEEDS \
    --n_scenarios $N_SCENARIOS \
    --loss_prob $LOSS_PROB \
    --margins $MARGIN \
    --loss_threshs $LOSS_THRESH \
    --Ms $M \
    --scales $SCALE \
    --recenter False \
    --sample_rho True \
    > "${OUTDIR}/log150_1.txt" 2>&1

echo "Run finished. Log saved to ${OUTDIR}/log150_1.txt"
