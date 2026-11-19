#!/bin/bash

# Example usage:
# bash run_tim_bayesian.sh
# source /nfs/home/dannis/anaconda3/envs/myenv/bin/python
source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv
which python

OUTPUT_FOLDER="20250923_true_mean_5e-5_large_sigma"  # Output directory for results
NTRADES_LIST="30 40 50 1000"          # Example trade sizes
NUM_RUNS=10              # Number of simulation runs
SEED=100
KAPPA=5e-5              # Transient impact scale
RHO=2.231               # Decay rate
GAMMA=5e-5              # Permanent impact scale
SIGMA=0.95              # Volatility
TAU=0.01                # Time step
MIN_GAP=100              # Minimum gap between trades
DRAWS=1000
T_MEAN_PRIOR=1e-4
T_STD_PRIOR=2e-4
ALPHA_A_PRIOR=2
ALPHA_B_PRIOR=2
RHO_MEAN_PRIOR=2.231
RHO_STD_PRIOR=20
# Call the Python script with arguments
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
    --draws $DRAWS \
    --t_mean_prior $T_MEAN_PRIOR \
    --t_std_prior $T_STD_PRIOR \
    --alpha_a_prior $ALPHA_A_PRIOR \
    --alpha_b_prior $ALPHA_B_PRIOR \
    --rho_mean_prior $RHO_MEAN_PRIOR \
    --rho_std_prior $RHO_STD_PRIOR \

echo "Bayesian MCMC finished."

