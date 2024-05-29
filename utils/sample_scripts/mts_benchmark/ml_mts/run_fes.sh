#!/bin/bash

# Define the number of windows to submit
NUM_WINDOWS=15
SCRIPT="run_fes"

# Loop to submit jobs
for i in $(seq 0 $NUM_WINDOWS); do
    # Create a unique job name for each submission
    WINDOW=$i

    OUTPUT_DIR=OUTPUT_$i
    mkdir $OUTPUT_DIR
    cd $OUTPUT_DIR
    echo "Submitting window $i"

    # Submit the job
    sbatch -J ${WINDOW}_${SCRIPT} ../submit_fes_window.slurm ${SCRIPT} ${WINDOW}

    cd ..
done
echo "All jobs have been submitted."
