#!/bin/bash

# Example usage:
# bash run_tim_opt.sh 20250620_162746

conda activate myenv

# Set optional arguments
OUTDIR="TIM_opt_results_20250628"
MCMC_TIMESTAMP="20250701_134415"
TRADE_LIST="1400"       # Example trade sizes
RUN_SEEDS=$(echo {122..129})
N_SCENARIOS=1000                # Example number of scenarios

mkdir -p $OUTDIR

# Call the Python script with arguments
python3 TIM/optimization.py \
    --outdir $OUTDIR \
    --mcmc_timestamp $MCMC_TIMESTAMP \
    --trade_list $TRADE_LIST \
    --run_seeds $RUN_SEEDS \
    --n_scenarios $N_SCENARIOS \
    | tee ${OUTDIR}/run_log_1400_0713_0.txt

echo "Run finished. Log saved to ${OUTDIR}/run_log_1400_0713_0.txt"