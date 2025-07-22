#!/bin/bash

# Example usage:
# bash run_tim_opt.sh 20250620_162746

conda activate myenv

# Set optional arguments
OUTDIR="TIM_fix_mean_results"
VAR_LIST="[[1.41e-6, 27.93], [1.91e-6, 9.24]]"      # Example trade sizes

RUN_SEEDS=$(echo {115..121})
N_SCENARIOS=1000                # Example number of scenarios

mkdir -p $OUTDIR

# Call the Python script with arguments
python3 TIM/optimization.py \
    --outdir $OUTDIR \
    --var_list "$VAR_LIST" \
    --run_seeds $RUN_SEEDS \
    --n_scenarios $N_SCENARIOS \
    | tee ${OUTDIR}/run_log_1100_1400_0713_1.txt

echo "Run finished. Log saved to ${OUTDIR}/run_log_1100_1400_0713_1.txt"