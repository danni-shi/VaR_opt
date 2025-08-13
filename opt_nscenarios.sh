#!/bin/bash

conda activate myenv

# Set optional arguments
OUTDIR="TIM_change_nscenarios_results"
VAR_LIST="1.91e-06 9.24"  
set -- $VAR_LIST
theta_v=$1
alpha_ab=$2

RUN_SEEDS=$(echo {1..5})

mkdir -p "$OUTDIR"

# Loop over N_SCENARIOS values
for N_SCENARIOS in $(seq 200 200 1200); do
    SUB_OUTDIR="${OUTDIR}/theta_${theta_v}_alpha_${alpha_ab}/nscen_${N_SCENARIOS}"
    echo "Creating subdirectory: $SUB_OUTDIR"
    mkdir -p "${SUB_OUTDIR}"

    # Call the Python script with arguments
    python3 TIM/optimization_cn.py \
        --outdir "$OUTDIR" \
        --var_list "$VAR_LIST" \
        --run_seeds "$RUN_SEEDS" \
        --n_scenarios "$N_SCENARIOS" \
        | tee "${SUB_OUTDIR}/log.txt"

    echo "Run with N_SCENARIOS=${N_SCENARIOS} finished. Log saved to ${SUB_OUTDIR}/log.txt"
done
