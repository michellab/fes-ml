# fes-ml

![Continuous Integration](https://github.com/michellab/fes-ml/actions/workflows/main.yaml/badge.svg)
![Test Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/JMorado/4e01061daef80d7212844cc9cd272a01/raw/fes_ml_pytest_coverage_report_main.json)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A package to run hybrid ML/MM free energy simulations.

## Table of Contents

1. [Installation](#installation)
2. [Alchemical Modifications](#alchemical-modifications)
3. [Running a Multistate Equilibrium Free Energy Simulation](#running-a-multistate-equilibrium-free-energy-simulation)
   1. [Using Multiple Alchemical Groups](#using-multiple-alchemical-groups)
4. [Dynamics and EMLE settings](#dynamics-and-emle-settings)
    1. [Sire Strategy](#sire-strategy)
    2. [OpenFF Strategy](#openff-strategy)
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
- `EMLEPotential`: Interpolate between electrostatic (`EMLEPotential=1`) and mechanical (`EMLEPotential=0`) embedding.
- `MLCorrection`: Interpolate between the ML (`MLCorrection=1`) and MM (`MLCorrection=0`) potentials through a Δ correction.
- `CustomLJ`: Modify the LJ parameters for interactions between the ML and MM systems, disregarding the specified lambda value.

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
fes.minimize(1000)
# Equilibrate all intermediate states for 1 ns
fes.equilibrate(1000000)
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

### Using Multiple Alchemical Groups

For more complex transformations, you can define multiple alchemical groups that can be transformed independently or simultaneously. This is particularly useful when you want to apply different transformations to different regions of your system or transform multiple ligands separately.

To use multiple alchemical groups, specify the group name as a suffix after a colon in the lambda schedule:

```python
from fes_ml.fes import FES
import numpy as np

# Define lambda schedule for multiple alchemical groups
lambda_schedule = {
    # Group 1: Turn off LJ and charges for ligand 1
    "LJSoftCore:ligand1": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0],
    "ChargeScaling:ligand1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0],

    # Group 2: Turn off LJ and charges for ligand 2
    "LJSoftCore:ligand2": [1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0],
    "ChargeScaling:ligand2": [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.33, 0.0],

    # Group 3: Interpolate between MM and ML for the entire system
    "MLInterpolation:system": [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
}

# Define atom indices for each alchemical group
ligand1_atoms = [1, 2, 3, 4, 5]      # Atoms belonging to first ligand
ligand2_atoms = [20, 21, 22, 23, 24] # Atoms belonging to second ligand
system_atoms = list(range(1, 50))     # All atoms for ML/MM interpolation

# Define per-group alchemical atoms and other settings
modifications_kwargs = {
    "LJSoftCore:ligand1": {
        "alchemical_atoms": ligand1_atoms
    },
    "ChargeScaling:ligand1": {
        "alchemical_atoms": ligand1_atoms
    },
    "LJSoftCore:ligand2": {
        "alchemical_atoms": ligand2_atoms
    },
    "ChargeScaling:ligand2": {
        "alchemical_atoms": ligand2_atoms
    },
    "MLInterpolation:system": {
        "alchemical_atoms": system_atoms
    }
}
```

#### Multiple Instances of the Same Modification Type

You can also use multiple instances of the same modification type for the same group groups. For example, to apply interpolate between two sets of `CustomLJ` parameters:

```python
# Lambda schedule with multiple CustomLJ modifications
lambda_schedule = {
    "LJSoftCore:openff1": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    "LJSoftCore:openff2": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "CustomLJ:openff1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "CustomLJ:openff2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

# Define different atoms and LJ parameters for each region
modifications_kwargs = {
    "CustomLJ:region1": {
        "lj_offxml": "openff_unconstrained-1.0.0.offxml",
    },
    "CustomLJ:region2": {
        "original_offxml": ["openff-2.1.0.offxml"],
        "lj_offxml": "openff_unconstrained-2.0.0.offxml",
    }
}
```

## Dynamics and EMLE settings

In fes-ml, the default strategy for creating OpenMM systems is through Sire. Additionally, fes-ml offers the OpenFF strategy. You can select the desired creation strategy, either `'sire'` or `'openff'`, using the `strategy_name` argument when calling the `fes.create_alchemical_states` method to create the alchemical systems. Most other simulation configurations can also be set by passing additional arguments to this method. For details on customization, refer to the definitions of the `SireCreationStrategy` and `OpenFFCreationStrategy` classes.

### Sire Strategy

Therefore, the options of the dynamics are modifiable and are the same as [those available in Sire](https://sire.openbiosim.org/cheatsheet/openmm.html#choosing-options). Typically, these are set in a `dynamics_kwargs` dictionary:

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

### OpenFF Strategy

In the OpenFF strategy, the dynamics options are also modifiable and can be set by passing a dictionary with the settings that will be used in the [`MDConfig` object](https://docs.openforcefield.org/projects/interchange/en/stable/_autosummary/openff.interchange.components.mdconfig.MDConfig.html#openff.interchange.components.mdconfig.MDConfig). The default `MDConfig` settings are as follows:

```python
from openff.units import unit as offunit

# This is the default mdconfig dictionary, meaning that if this dictionary
# is not passed to the FES object, these are the values that will be used.
mdconfig_dict = {
    "periodic": True,
    "constraints": "h-bonds",
    "vdw_method": "cutoff",
    "vdw_cutoff": offunit.Quantity(12.0, "angstrom"),
    "mixing_rule": "lorentz-berthelot",
    "switching_function": True,
    "switching_distance": offunit.Quantity(11.0, "angstrom"),
    "coul_method": "pme",
    "coul_cutoff": offunit.Quantity(12.0, "angstrom"),
}
```

The solvation options can also be set by passing a `packmol_kwargs` dictionary to `fes.create_alchemical_states`:

```python
from openff.interchange.components._packmol import UNIT_CUBE

packmol_kwargs = {
    "box_shape": UNIT_CUBE,
    "mass_density": 0.779 * offunit.gram / offunit.milliliter
}
```

The customizable options can be checked [here](https://github.com/openforcefield/openff-interchange/blob/main/openff/interchange/components/_packmol.py#L564-L574). For example, the above density can be used to create a ligand solvated in a cubic cyclohexane box.

## Logging Settings

By default, fes-ml logs messages at the `INFO` level. This means you will see informative messages about the overall progress but not necessarily detailed debugging information. You can control the verbosity of the logging output by setting the `FES_ML_LOG_LEVEL` environment variable:

```bash
export FES_ML_LOG_LEVEL="DEBUG"
```

If you want to include log messages from packages other than fes-ml, set the `FES_ML_FILTER_LOGGERS` variable to 0:

```bash
export FES_ML_FILTER_LOGGERS=0
```

By default, this variable is set to 1, meaning only log messages coming from `fes-ml` are displayed.

If you want, for debugging purposes, to report the energy decomposition of each created alchemical state before and after the alchemical modification, set the `FES_ML_LOG_DEVEL` variable to 1:

```bash
export FES_ML_LOG_DEVEL=1
```

Note that reporting the energy decomposition is disabled by default in fes-ml, as it is an expensive operation, especially if ML potentials are present, due to the need to reinitialize the context.
