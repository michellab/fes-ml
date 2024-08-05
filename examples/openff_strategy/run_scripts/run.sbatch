#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fes-ml
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/fes-ml_%A.out

date
start=`date +%s`

export scripts_dir=$(pwd)

# TODO activate fes ml environment
export env="/home/anna/anaconda3/bin/activate /home/anna/mambaforge/envs/fes-ml"
source $env

export FES_ML_LOG_LEVEL="DEBUG"

run_dir=test
smiles="c1ccccc1" # $1
#smiles="c1ccc2c(c1)C(=O)c3c(ccc(c3C2=O)N)N"

python $scripts_dir/create_run_system.py --smiles $smiles --folder $run_dir --timestep 2 --use-mts --inner-steps 4 --intermediate-steps 2 # --use-hmr

python $scripts_dir/agg_ouput.py --folder $run_dir

python $scripts_dir/analyse.py --folder $run_dir

echo "done."

end=`date +%s`
runtime=$((end-start))
echo "runtime was $runtime seconds, which is $((runtime/60)) minutes"