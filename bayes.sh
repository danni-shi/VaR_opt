#!/bin/bash

# Example usage:
# bash run_tim_bayesian.sh
# source /nfs/home/dannis/anaconda3/envs/myenv/bin/python
source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv
which python

# Set optional arguments
OUTPUT_FOLDER="20250830"  # Output directory for results
TRADE_LIST="1000"         # Example trade sizes
NUM_RUNS=20              # Number of simulation runs
KAPPA=1e-4              # Transient impact scale
RHO=2.231               # Decay rate
GAMMA=1e-4              # Permanent impact scale
SIGMA=0.95              # Volatility
TAU=0.01                # Time step
MIN_GAP=100              # Minimum gap between trades


# Call the Python script with arguments
python3 TIM/bayesian.py \
    --output_folder $OUTPUT_FOLDER \
    --trade_list $TRADE_LIST \
    --num_runs $NUM_RUNS \
    --kappa $KAPPA \
    --rho $RHO \
    --gamma $GAMMA \
    --sigma $SIGMA \
    --tau $TAU \
    --min_gap $MIN_GAP

echo "Bayesian MCMC finished."