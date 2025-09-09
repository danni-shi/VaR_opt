#!/bin/bash

# Example usage:
# bash run_tim_opt.sh 20250620_162746

source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Set optional arguments
OUTDIR="TIM_opt_results_20250909_thresh10_n3000"
MCMC_TIMESTAMP="20250830"
TRADE_LIST="400 1000"    # Example trade sizes
RUN_SEEDS=$(echo {100..119})
N_SCENARIOS=3000                # Example number of scenarios
LOSS_PROB=0.1                  # Example loss probability
MARGIN=0.0                    # Example margin
mkdir -p $OUTDIR

# Call the Python script with arguments
python3 TIM/optimization.py \
    --outdir $OUTDIR \
    --mcmc_timestamp $MCMC_TIMESTAMP \
    --trade_list $TRADE_LIST \
    --run_seeds $RUN_SEEDS \
    --n_scenarios $N_SCENARIOS \
    --loss_prob $LOSS_PROB \
    --margin $MARGIN \
    > "${OUTDIR}/log.txt" 2>&1

echo "Run finished. Log saved to ${OUTDIR}/log.txt"