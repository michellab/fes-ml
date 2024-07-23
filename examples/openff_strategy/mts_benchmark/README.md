## Running instructions

1. For example inside the `mm` directory, run the simulations by executing the following commands (just adapt `submit_fes_window.slurm` to your system/HPC cluster):

```bash
chmod 750 ./run_fes.sh
./run_fes.sh
```

This command submits one job per window.

2. Once all simulations have finished, merge the output from each one:

```bash
python agg_output.py
```

This will create a file named `U_kln_agg.npy` containing the aggregated data.

3. Calculate the free energy using PyMBAR:

```bash
python analyse.py U_kln_agg.npy 298.15
```

Here, the first positional argument is the name of the input file, and the second is the temperature in Kelvin.
