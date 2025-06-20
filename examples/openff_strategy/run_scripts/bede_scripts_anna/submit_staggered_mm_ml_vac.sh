#!/bin/bash

if [[ ! -d slurm_logs ]]; then
    mkdir slurm_logs
fi

export FES_ML_LOG_LEVEL="INFO"
export FES_ML_FILTER_LOGGERS=0
export FES_ML_DEBUG_LEVEL=1

NUM_WINDOWS=6

for w in $(seq 0 $NUM_WINDOWS); do

# Make a list of the transformations from the scheme file 
settings_file="settings.dat"
scheme_array=()
dt_array=()
intermediate_array=()
inner_array=()
hmr_array=()
IFS=','
while read lines; do
while read -a line; do scheme=${line[0]}; dt=${line[1]}; int=${line[2]}; inner=${line[3]}; hmr=(${line[-1]});
scheme_array+=("$scheme"); dt_array+=("$dt"); intermediate_array+=("$int"); inner_array+=("$inner"); hmr_array+=("$hmr"); done <<< $lines
done < $settings_file

ligand_array=( "mobley_3053621" "mobley_242480" )
smiles_array=( "c1ccccc1" "c1ccc2c(c1)C(=O)c3c(ccc(c3C2=O)N)N" )

for l in ${!ligand_array[@]}; do
ligand=${ligand_array[l]}
smiles=${smiles_array[l]}

echo $ligand, $smiles

if [[ ! -d $ligand ]]; then
    mkdir $ligand
fi

for i in ${!scheme_array[@]}; do

if [[ ! -d $ligand/${scheme_array[i]} ]]; then
    mkdir $ligand/${scheme_array[i]}
fi

if [[ ! -d $ligand/${scheme_array[i]}/mm_ml_vac ]]; then
    mkdir $ligand/${scheme_array[i]}/mm_ml_vac
fi

arguments=( --smiles $smiles --folder $ligand/${scheme_array[i]}/mm_ml_vac )
arguments+=( --timestep ${dt_array[i]} )
if [ ${inner_array[i]} ] ; then
arguments+=( --use-mts --inner-steps ${inner_array[i]} )
fi
if [ ${intermediate_array[i]} ] ; then
arguments+=( --intermediate-steps ${intermediate_array[i]} )
fi
if [ ${hmr_array[i]} == "True" ] ; then
arguments+=( --use-hmr )
fi
#if [ ${nonbonded_array[i]} == "True" ] ; then
#arguments+=( --do-not-split-nonbonded )
#fi
arguments+=( --windows $NUM_WINDOWS --sampling 6 --do-not-vanish-ligand --ml-application correction --vacuum --run_window $w )

echo " ${arguments[@]} "
array_string=$(IFS=,; echo "${arguments[*]}")
sbatch submit_fes_window.sbatch "$array_string"

done
echo ""
done

done

echo "All jobs have been submitted."