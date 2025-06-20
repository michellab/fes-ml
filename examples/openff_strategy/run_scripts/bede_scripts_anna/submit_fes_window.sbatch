#!/bin/bash
#SBATCH --account=bdhbs30
#SBATCH --time=48:00:00       # Run for a max of 1 hour
#SBATCH --partition=gpu    # gpu for BEDE
#SBATCH --nodes=1           # Resources from a single node
#SBATCH --gres=gpu:1        # One GPU per node (plus 25% of node CPU and RAM per GPU)
#SBATCH -o slurm_logs/fes-ml_%A.out


date
start=`date +%s`

IFS=',' read -r -a arguments <<< "$1"
echo ${arguments[@]}
echo ${arguments[3]}

python create_run_system_single_window.py ${arguments[@]}

python agg_ouput.py --folder ${arguments[3]}

python analysis.py --folder ${arguments[3]}

echo "done."

end=`date +%s`
runtime=$((end-start))
echo "runtime was $runtime seconds, which is $((runtime/60)) minutes"
