#!/bin/bash

# Example usage:
# bash run_tim_opt.sh 20250620_162746

source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Set optional arguments
OUTDIR="TIM_opt_results/20250909_vary_M"
MCMC_TIMESTAMP="20250830"
TRADE_LIST="50 150 400 1000"    # Example trade sizes
RUN_SEEDS=$(echo {100..119})
N_SCENARIOS=2000                # Example number of scenarios
LOSS_PROB=0.1                  # Example loss probability
MARGIN=0.0                    # Example margin

# Define different values of M to test
M_VALUES="1e5 5e5 1e6 5e6 1e7"

mkdir -p $OUTDIR

# Loop through different M values
for M in $M_VALUES; do
    echo "Running optimization with M=$M"
    
    # Call the Python script with arguments
    python3 TIM/optimization.py \
        --outdir $OUTDIR \
        --mcmc_timestamp $MCMC_TIMESTAMP \
        --trade_list $TRADE_LIST \
        --run_seeds $RUN_SEEDS \
        --n_scenarios $N_SCENARIOS \
        --loss_prob $LOSS_PROB \
        --margin $MARGIN \
        --M $M \
        > "${OUTDIR}/log_M_${M}.txt" 2>&1
    
    echo "Finished M=$M. Log saved to ${OUTDIR}/log_M_${M}.txt"
done

echo "All runs finished."