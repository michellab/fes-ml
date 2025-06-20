#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fes-ml
#SBATCH --time=30-00:00:00
#SBATCH -o slurm_logs/fes-ml_%A.out

date
start=`date +%s`

export scripts_dir=/home/anna/Documents/multitimestep/fes-ml/examples/openff_strategy/run_scripts

# TODO activate fes ml environment
export env="/home/anna/anaconda3/bin/activate /home/anna/mambaforge/envs/fes-ml"
source $env


export FES_ML_LOG_LEVEL="INFO"
export FES_ML_FILTER_LOGGERS=0
export FES_ML_DEBUG_LEVEL=1
export INTERCHANGE_EXPERIMENTAL=1

IFS=',' read -r -a arguments <<< "$1"
echo ${arguments[@]}
echo ${arguments[3]}

python create_run_system.py ${arguments[@]}

python $scripts_dir/agg_ouput.py --folder ${arguments[3]}

python $scripts_dir/analysis.py --folder ${arguments[3]}

echo "done."

end=`date +%s`
runtime=$((end-start))
echo "runtime was $runtime seconds, which is $((runtime/60)) minutes"
