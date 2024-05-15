#!/bin/bash 
#SBATCH -p main 
#SBATCH -n 1
#SBATCH --gres=gpu:1 
#SBATCH --time 1-00:00:00

# Capture first positional argumnet
SCRIPT=$1
WINDOW=$2

# Export environmental variables
python -u ../${SCRIPT}.py ${SCRIPT} ${WINDOW}> ${SCRIPT}_${WINDOW}.output 2> ${SCRIPT}_${WINDOW}.errors 
