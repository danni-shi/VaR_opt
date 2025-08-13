#!/bin/bash

conda activate myenv

# Set optional arguments
OUTDIR="TIM_fix_mean_results"
VAR_LIST="1.91e-06 9.24"  
set -- $VAR_LIST
theta_v=$1
alpha_ab=$2

SUB_OUTDIR="${OUTDIR}/theta_${theta_v}_alpha_${alpha_ab}"
echo "Creating subdirectory: $SUB_OUTDIR"
RUN_SEEDS=$(echo {1..10})
N_SCENARIOS=1000                # Example number of scenarios

mkdir -p $OUTDIR
mkdir -p "${SUB_OUTDIR}"

# Call the Python script with arguments
python3 TIM/optimization_cn.py \
    --outdir $OUTDIR \
    --var_list $VAR_LIST \
    --run_seeds $RUN_SEEDS \
    --n_scenarios $N_SCENARIOS \
    | tee ${SUB_OUTDIR}/log.txt

echo "Run finished. Log saved to ${SUB_OUTDIR}/log.txt"
