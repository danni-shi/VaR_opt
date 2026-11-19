#!/bin/bash

# Example usage:
# bash opt_vary_mean.sh

source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Define the different MEAN values to test
MEAN_VALUES="5.05e-5"

# Loop through each MEAN value
for MEAN in $MEAN_VALUES; do
    echo "Running optimization with MEAN=$MEAN"
    
    # Set optional arguments
    OUTDIR="/nfs/home/dannis/AC-model/TIM_opt_results/20250929_vary_mean_${MEAN}"
    MCMC_TIMESTAMP="20250916_online_5e-5"
    TRUE_GAMMA=$MEAN
    TRUE_KAPPA=$MEAN
    LIST_SIGMA=0.95  
    LIST_N_TRADES="150"    # Example trade sizes
    MCMC_LABELS="combined"
    RUN_SEEDS=$(echo {100..105})
    N_SCENARIOS="1000"               # Example number of scenarios
    LOSS_PROB=0.1                  # Example loss probability
    MARGIN="0.05"                    # Example margin
    LOSS_THRESH="4.5e5"  # Example loss thresholds
    M="1e6"          # Example M values
    SCALE="100"                     # Example scale factor
    mkdir -p $OUTDIR

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
        --recenter True \
        --sample_rho False \
        > "${OUTDIR}/log.txt" 2>&1

    echo "Run finished for MEAN=$MEAN. Log saved to ${OUTDIR}/log.txt"
done

echo "All runs completed!"