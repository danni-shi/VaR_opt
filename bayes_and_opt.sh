#!/bin/bash

# Example usage:
# bash run_tim_bayesian.sh
# source /nfs/home/dannis/anaconda3/envs/myenv/bin/python
source /nfs/home/dannis/anaconda3/etc/profile.d/conda.sh
conda activate myenv
which python

# Set optional arguments
OUTPUT_FOLDER="20250916_online_5e-5"  # Output directory for results
NTRADES_LIST="60"          # Example trade sizes
NUM_RUNS=20              # Number of simulation runs
SEED=100
KAPPA=5e-5              # Transient impact scale
RHO=2.231               # Decay rate
GAMMA=5e-5              # Permanent impact scale
SIGMA=0.95              # Volatility
TAU=0.01                # Time step
MIN_GAP=100              # Minimum gap between trades
DRAWS=1000
T_MEAN_PRIOR=5e-4
T_STD_PRIOR=2e-4
ALPHA_A_PRIOR=2
ALPHA_B_PRIOR=1
RHO_MEAN_PRIOR=8
RHO_STD_PRIOR=2.0

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

# Set optional arguments
OUTDIR="/nfs/home/dannis/AC-model/TIM_opt_results/20250922_recenter_posteriors_5e-5"
MCMC_TIMESTAMP=$OUTPUT_FOLDER
TRUE_GAMMA=$GAMMA
TRUE_KAPPA=$KAPPA
LIST_N_TRADES=$NTRADES_LIST    # Example trade sizes
MCMC_LABELS="combined"
RUN_SEEDS=$(echo {100..109})
N_SCENARIOS="1000"               # Example number of scenarios
LOSS_PROB=0.1                  # Example loss probability
MARGIN="0.05"                    # Example margin
LOSS_THRESH="4.5e5"  # Example loss thresholds
M="1e6"          # Example M values
SCALE="100"                     # Example scale factor
LIST_SIGMA="0.95"
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
    --use_info_file False \
    > "${OUTDIR}/log.txt" 2>&1

echo "Run finished. Log saved to ${OUTDIR}/log.txt"

OUTDIR="/nfs/home/dannis/AC-model/TIM_opt_results/20250930_5e-5"

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
    --recenter False \
    --sample_rho True \
    --use_info_file False \
    > "${OUTDIR}/log.txt" 2>&1

echo "Run finished. Log saved to ${OUTDIR}/log.txt"