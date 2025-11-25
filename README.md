# fes-ml

![Continuous Integration](https://github.com/michellab/fes-ml/actions/workflows/main.yaml/badge.svg)
![Test Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/JMorado/4e01061daef80d7212844cc9cd272a01/raw/fes_ml_pytest_coverage_report_main.json)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python package for running hybrid machine learning/molecular mechanics (ML/MM) free energy simulations using alchemical transformations.

## Table of Contents

1. [Installation](#installation)
2. [Alchemical Modifications](#alchemical-modifications)
3. [Running a Multistate Equilibrium Free Energy Simulation](#running-a-multistate-equilibrium-free-energy-simulation)
   - [Using Multiple Alchemical Groups](#using-multiple-alchemical-groups)
4. [Dynamics and EMLE Settings](#dynamics-and-emle-settings)
   - [Sire Strategy](#sire-strategy)
   - [OpenFF Strategy](#openff-strategy)
5. [Logging Settings](#logging-settings)

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

The following alchemical transformations are supported in fes-ml:

- `LJSoftCore`: Turn on (`LJSoftCore=1`) or off (`LJSoftCore=0`) Lennard-Jones 12-6 interactions using a softcore potential.
- `ChargeScaling`: Turn on (`ChargeScaling=1`) or off (`ChargeScaling=0`) electrostatic interactions by scaling atomic charges.
- `MLInterpolation`: Interpolate between ML (`MLInterpolation=1`) and MM (`MLInterpolation=0`) potentials.
- `EMLEPotential`: Interpolate between electrostatic (`EMLEPotential=1`) and mechanical (`EMLEPotential=0`) embedding.
- `MLCorrection`: Interpolate between ML (`MLCorrection=1`) and MM (`MLCorrection=0`) potentials through a Δ-learning correction.
- `CustomLJ`: Modify LJ parameters for interactions between ML and MM systems. When interpolating between two sets of LJ parameters, `CustomLJ=1` corresponds to the modified LJ parameters and `CustomLJ=0` to the original ones.

The lambda schedule is defined using a dictionary. For example, to turn off the LJ 12-6 interactions in steps of 0.2, followed by turning off charges in steps of 0.33, use the following lambda schedule:

```python
lambda_schedule = {
    "LJSoftCore": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0],
    "ChargeScaling": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.66, 0.33, 0.00]
}
```

## Running a Multistate Equilibrium Free Energy Simulation

Currently, only equilibrium free energy simulations are supported. Non-equilibrium methods are not yet implemented.

Once a lambda schedule has been defined, the free energy calculation can be run using the following script, which executes all intermediate states in serial:

```python
from fes_ml.fes import FES
import numpy as np

# Create the FES object to run the simulations
fes = FES()

# List of atom indices to alchemify
alchemical_atoms = [1, 2, 3]

# Create the alchemical states
fes.create_alchemical_states(
    top_file="path_to_topology_file",
    crd_file="path_to_coordinates_file",
    alchemical_atoms=alchemical_atoms,
    lambda_schedule=lambda_schedule,
)

# Minimize all intermediate states (1000 steps)
fes.minimize(1000)

# Equilibrate all intermediate states for 1 ns
fes.equilibrate(1000000)

# Sample 1000 times every ps (i.e., 1 ns of simulation per state)
U_kln = fes.run_production_batch(1000, 1000)

# Save the data for analysis
np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
```

Alternatively, individual intermediate states can be run separately, enabling parallelization across multiple scripts. For example, to run window 6:

```python
# Sample 1000 times every ps (i.e., 1 ns of simulation for state 6)
U_kln = fes.run_single_state(1000, 1000, 6)

# Save the data for analysis
np.save("U_kln_mm_sol_6.npy", np.asarray(U_kln))
```

### Using Multiple Alchemical Groups

For more complex transformations, you can define multiple alchemical groups that can be transformed independently or simultaneously. This is particularly useful when applying different transformations to different regions of your system or when transforming multiple ligands separately.

To use multiple alchemical groups, specify the group name as a suffix after a colon in the lambda schedule keys:

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
ligand1_atoms = [1, 2, 3, 4, 5]       # Atoms belonging to first ligand
ligand2_atoms = [20, 21, 22, 23, 24]  # Atoms belonging to second ligand
system_atoms = list(range(1, 50))     # All atoms for ML/MM interpolation

# Define per-group alchemical atoms
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

## Dynamics and EMLE Settings

In fes-ml, the default strategy for creating OpenMM systems is through [Sire](https://sire.openbiosim.org). Additionally, fes-ml supports the OpenFF strategy. You can select the desired creation strategy (`'sire'` or `'openff'`) using the `strategy_name` argument when calling the `fes.create_alchemical_states` method. Most simulation configurations can also be customized by passing additional arguments to this method. For details on available options, refer to the definitions of the `SireCreationStrategy` and `OpenFFCreationStrategy` classes.

### Sire Strategy

The dynamics options are modifiable and follow the same conventions as [those available in Sire](https://sire.openbiosim.org/cheatsheet/openmm.html#choosing-options). Typically, these are set in a `dynamics_kwargs` dictionary:

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

Similarly, keyword arguments passed to the [`EMLECalculator`](https://github.com/chemle/emle-engine/blob/main/emle/calculator.py#L315) can also be configured:

```python
# Define the parameters for the EMLE potential
emle_kwargs = {"method": "electrostatic", "backend": "torchani", "device": "cpu"}
```

These dictionaries can then be passed when creating the alchemical states:

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

In the OpenFF strategy, dynamics options are also modifiable and can be set by passing a dictionary with settings used in the [`MDConfig` object](https://docs.openforcefield.org/projects/interchange/en/stable/_autosummary/openff.interchange.components.mdconfig.MDConfig.html#openff.interchange.components.mdconfig.MDConfig). The default `MDConfig` settings are:

```python
from openff.units import unit as offunit

# These are the default mdconfig settings. If this dictionary is not passed
# to the FES object, these values will be used automatically.
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

Solvation options can also be set by passing a `packmol_kwargs` dictionary to `fes.create_alchemical_states`:

```python
from openff.interchange.components._packmol import UNIT_CUBE

packmol_kwargs = {
    "box_shape": UNIT_CUBE,
    "mass_density": 0.779 * offunit.gram / offunit.milliliter
}
```

The customizable options can be found [here](https://github.com/openforcefield/openff-interchange/blob/main/openff/interchange/components/_packmol.py#L564-L574). For example, the density above can be used to create a ligand solvated in a cubic cyclohexane box.

## Logging Settings

By default, fes-ml logs messages at the `INFO` level, showing informative messages about overall progress without detailed debugging information. You can control the logging verbosity by setting the `FES_ML_LOG_LEVEL` environment variable:

```bash
export FES_ML_LOG_LEVEL="DEBUG"
```

To include log messages from packages other than fes-ml, set the `FES_ML_FILTER_LOGGERS` variable to 0:

```bash
export FES_ML_FILTER_LOGGERS=0
```

By default, this variable is set to 1, filtering out log messages from other packages.

For debugging purposes, you can report the energy decomposition of each alchemical state before and after the alchemical modification by setting the `FES_ML_LOG_DEVEL` variable to 1:

```bash
export FES_ML_LOG_DEVEL=1
```

Note that energy decomposition reporting is disabled by default because it is an expensive operation, especially with ML potentials present, due to the need to reinitialize the context.
