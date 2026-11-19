#!/bin/bash

# Example usage:
# bash run_tim_bayesian.sh
# source /nfs/home/dannis/anaconda3/envs/myenv/bin/python
source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv
which python

# Set optional arguments
OUTPUT_FOLDER="20251014_open_and_close"  # Output directory for results
NDAYS_LIST="30 40 50 60 80 150 250 400 600 800 1000 1500"          # Example trade sizes
NUM_RUNS=20              # Number of simulation runs
DATA_METHOD="gaps_poisson"  # Method for simulating trade times
SEED=100
KAPPA=5e-5              # Transient impact scale
RHO=2.231               # Decay rate
GAMMA=5e-5              # Permanent impact scale
SIGMA=0.95              # Volatility
TAU=0.01                # Time step
MIN_GAP=100              # Minimum gap between trades
DRAWS=1000
TUNE=4000
T_MEAN_PRIOR=5e-4
T_STD_PRIOR=2e-4
ALPHA_A_PRIOR=2
ALPHA_B_PRIOR=1
RHO_MEAN_PRIOR=8.0
RHO_STD_PRIOR=2.0
# Call the Python script with arguments
python3 TIM/bayesian_by_day.py \
    --output_folder $OUTPUT_FOLDER \
    --ndays_list $NDAYS_LIST \
    --num_runs $NUM_RUNS \
    --seed $SEED \
    --kappa $KAPPA \
    --rho $RHO \
    --gamma $GAMMA \
    --sigma $SIGMA \
    --tau $TAU \
    --draws $DRAWS \
    --tune $TUNE \
    --t_mean_prior $T_MEAN_PRIOR \
    --t_std_prior $T_STD_PRIOR \
    --alpha_a_prior $ALPHA_A_PRIOR \
    --alpha_b_prior $ALPHA_B_PRIOR \
    --rho_mean_prior $RHO_MEAN_PRIOR \
    --rho_std_prior $RHO_STD_PRIOR \

echo "Bayesian MCMC finished."