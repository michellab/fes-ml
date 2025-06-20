#!/bin/bash

if [[ ! -d slurm_logs ]]; then
    mkdir slurm_logs
fi

# Make a list of the transformations from the scheme file 
settings_file="settings_mm_sol.dat"
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

# mobley_3053621 c1ccccc1
# mobley_20524 c1ccc(cc1)O
# mobley_2518989 CCOP(=S)(OCC)S[C@@H](CCl)N1C(=O)c2ccccc2C1=O
# mobley_9571888 C1[C@@H]2[C@H](COS(=O)O1)[C@@]3(C(=C([C@]2(C3(Cl)Cl)Cl)Cl)Cl)Cl
# mobley_1527293 C[C@@H](c1ccc(c(c1)F)c2ccccc2)C(=O)O
# mobley_242480 c1ccc2c(c1)C(=O)c3c(ccc(c3C2=O)N)N 

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

if [[ ! -d $ligand/${scheme_array[i]}/mm_ml_sol ]]; then
    mkdir $ligand/${scheme_array[i]}/mm_ml_sol
fi

arguments=( --smiles $smiles --folder $ligand/${scheme_array[i]}/mm_ml_sol )
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
arguments+=( --windows 6 --sampling 1.1 --do-not-vanish-ligand --ml-application correction  )

echo " ${arguments[@]} "
array_string=$(IFS=,; echo "${arguments[*]}")
sbatch run.sbatch "$array_string"

done
echo ""
done
