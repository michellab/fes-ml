# fes-ml

![Continuous Integration](https://github.com/michellab/fes-ml/actions/workflows/main.yaml/badge.svg)
![Test Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/JMorado/4e01061daef80d7212844cc9cd272a01/raw/fes_ml_pytest_coverage_report_main.json)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A package to run hybrid ML/MM free energy simulations.

## Table of Contents

1. [Installation](#installation)
2. [Alchemical Modifications](#alchemical-modifications)
3. [Running a Multistate Equilibrium Free Energy Simulation](#running-a-multistate-equilibrium-free-energy-simulation)
4. [Dynamics and EMLE settings](#dynamics-and-emle-settings)
5. [Log Level](#log-level)

## Installation

First, create a conda environment with all of the required dependencies:

```bash
conda env create -f environment.yaml
conda activate fes-ml
```

Finally, install `fes-ml` in interactive mode within the activated environment:

```bash
pip install -e .
```

## Alchemical Modifications

The following alchemical transformations can be performed in fes-ml:

- `LJSoftCore`: Turn on (`LJSoftCore=1`) and off (`LJSoftCore=0`) the Lennard-Jones 12-6 interactions by using a softcore potential.
- `ChargeScaling`: Turn on (`ChargeScaling=1`) and off (`ChargeScaling=0`) the electrostatic interactions by scaling the charges.
- `MLInterpolation`: Interpolate between the ML (`MLInterpolation=1`) and MM (`MLInterpolation=0`) potentials.
- `EMLEPotential`: Interpolate between the EMLE (`EMLEPotential=1`) and MM (`EMLEPotential=0`) potentials.
- `MLCorrection`: Interpolate between the ML (`MLCorrection=1`) and MM (`MLCorrection=0`) potentials through a Δ correction.

The lambda schedule to follow during the simulation is set in a dictionary. For example, to turn off the LJ 12-6 interactions in steps of 0.2 and subsequently turn off the charge in steps of 0.33, the following lambda schedule can be defined:

```python
lambda_schedule = {
    "LJSoftCore": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ChargeScaling": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.66, 0.33, 0.00]
}
```

## Running a Multistate Equilibrium Free Energy Simulation

Currently, only equilibrium free energy simulations are supported, meaning that non-equilibrium methods are not implemented yet.

Once a lambda schedule has been defined, the free energy calculation can be run using a script like this, which runs all intermediate states in serial:

```python
from fes_ml.fes import FES
import numpy as np

# Create the FES object to run the simulations
fes = FES()

# List with indexes of atoms to alchemify
alchemical_atoms = [1, 2, 3]

# Create the alchemical states
fes.create_alchemical_states(
    top_file="path_to_topology_file",
    crd_file="path_to_coordinates_file",
    alchemical_atoms=alchemical_atoms,
    lambda_schedule=lambda_schedule,
)

# Minimize all intermediate states
fes.minimize_batch(1000)
# Equilibrate all intermediate states for 1 ns
fes.equilibrate_batch(1000000)
# Sample 1000 times every ps (i.e., 1 ns of simulation per state)
U_kln = fes.run_production_batch(1000, 1000)
# Save the data to be analysed
np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
```

Alternatively, only one intermediate can be run at a time, allowing for easy parallelisation of the calculations by concurrently running multiple scripts. For example, to run window 6, use the following commands:

```python
# Sample 1000 times every ps (i.e., 1 ns of simulation per state)
U_kln = fes.run_single_state(1000, 1000, 6)

# Save the data to be analyzed
np.save("U_kln_mm_sol_6.npy", np.asarray(U_kln))
```

## Dynamics and EMLE settings

In fes-ml, the default strategy to create OpenMM systems is through Sire. Therefore, the options of the dynamics are modifiable and are the same as [those available in Sire](https://sire.openbiosim.org/cheatsheet/openmm.html#choosing-options). Typically, these are set in a `dynamics_kwargs` dictionary:

```python
# Define the dynamics parameters
dynamics_kwargs = {
    "timestep": "1fs",
    "cutoff_type": "PME",
    "cutoff": "12A",
    "constraint": "h_bonds",
    "integrator": "langevin_middle",
    "temperature": "298.15K",
    "pressure": "1atm",
    "platform": "cuda",
    "map": {"use_dispersion_correction": True, "tolerance": 0.0005},
}
```

Likewise, the keyword arguments passed to the [`EMLECalculator`](https://github.com/chemle/emle-engine/blob/main/emle/calculator.py#L315) can also be set:

```python
# Define the parameters of the EMLE potential
emle_kwargs = {"method": "electrostatic", "backend": "torchani", "device": "cpu"}
```

These dictionaries can then be passed upon creation of the alchemical states, i.e.:

```python
# Create the alchemical states
fes.create_alchemical_states(
    alchemical_atoms=alchemical_atoms,
    lambda_schedule=lambda_schedule,
    dynamics_kwargs=dynamics_kwargs,
    emle_kwargs=emle_kwargs
)
```

## Logging Settings

By default, fes-ml logs messages at the INFO level. This means you will see informative messages about the overall progress but not necessarily detailed debugging information. You can control the verbosity of the logging output by setting the FES_ML_LOG_LEVEL environment variable:

```bash
export FES_ML_LOG_LEVEL="DEBUG"
```

If you want to include log messages from packages other than fes-ml, set the `FES_ML_FILTER_LOGGERS` variable to 0:

```bash
export FES_ML_FILTER_LOGGERS=0
```

By default, this variable is set to 1, meaning only log messages coming from `fes-ml` are displayed.
