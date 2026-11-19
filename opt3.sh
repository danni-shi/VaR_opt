#!/bin/bash

# Example usage:
# bash run_tim_opt.sh 20250620_162746

source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Set optional arguments
OUTDIR="/nfs/home/dannis/AC-model/TIM_opt_results/20250929_1e-5"
MCMC_TIMESTAMP="20250918_online_1e-5"  # Example timestamp
LIST_N_TRADES="150 99999"    # Example trade sizes
MCMC_LABELS="combined"
TRUE_GAMMA=1e-5
TRUE_KAPPA=1e-5
RUN_SEEDS=$(echo {100..104})
N_SCENARIOS="1000"               # Example number of scenarios
LOSS_PROB=1.                  # Example loss probability
MARGIN="0.0"                    # Example margin
LOSS_THRESH="1e5"  # Example loss thresholds
M="5e5"          # Example M values
SCALE="100"                     # Example scale factor
LIST_SIGMA="0.45"

mkdir -p $OUTDIR

# Call the Python script with arguments
python3 TIM/optimization.py \
    --outdir $OUTDIR \
    --mcmc_timestamp $MCMC_TIMESTAMP \
    --list_n_trades $LIST_N_TRADES \
    --mcmc_labels $MCMC_LABELS \
    --true_gamma $TRUE_GAMMA \
    --true_kappa $TRUE_KAPPA \
    --run_seeds $RUN_SEEDS \
    --n_scenarios $N_SCENARIOS \
    --loss_prob $LOSS_PROB \
    --margins $MARGIN \
    --loss_threshs $LOSS_THRESH \
    --Ms $M \
    --scales $SCALE \
    --list_sigma $LIST_SIGMA \
    --recenter True \
    --use_info_file False \
    --sample_rho False \
    > "${OUTDIR}/log.txt" 2>&1

echo "Run finished. Log saved to ${OUTDIR}/log.txt"