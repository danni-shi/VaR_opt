#!/bin/bash

# Example usage:
# bash run_tim_bayesian.sh
# source /nfs/home/dannis/anaconda3/envs/myenv/bin/python

source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv
which python

# Set optional arguments
OUTPUT_FOLDER="20250916_online"  # Output directory for results
NTRADES_LIST="50 150 250 400 600 800 1000 1500 2000"  # Example trade sizes
NUM_RUNS=1
KAPPA=1e-5
RHO=2.231
GAMMA=1e-5
SIGMA=0.45
TAU=0.01
MIN_GAP=100
DRAWS=1000

# Loop over SEED values and run in parallel
for SEED in {110..119}; do
    echo "Launching run with SEED=$SEED"
    python3 TIM/bayesian.py \
        --output_folder $OUTPUT_FOLDER \
        --ntrades_list $NTRADES_LIST \
        --num_runs $NUM_RUNS \
        --seed $SEED \
        --kappa $KAPPA \
        --rho $RHO \
        --gamma $GAMMA \
        --sigma $SIGMA \
        --tau $TAU \
        --min_gap $MIN_GAP \
        --draws $DRAWS &
done

# Wait for all background jobs to finish before exiting
wait

echo "All Bayesian MCMC runs finished."
