"""
Script to run a performance benchmark of the FES-ML code using the OpenFF strategy.
The modifications used are MLCorrection and MLInterpolation. 
Only 1 window is run for 1 ns to test the performance of the code.

Authors: Joao Morado
"""

if __name__ == "__main__":
    import sys

    import numpy as np
    import openff.units as offunit
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from fes_ml import FES, MTS

    if len(sys.argv) != 1:
        raise ValueError("must pass window as positional arguments")

    window = int(sys.argv[1])

    print(f"Window : {window}")

    # --------------------------------------------------------------- #
    # Define the parameters
    # --------------------------------------------------------------- #
    # Set up the alchemical modifications
    lambda_schedule = {
        "MLInterpolation": [1.0],
        "EMLEPotential": [1.0],
    }

    # Modifications kwargs
    # This dictionary is used to pass additional kwargs to the modifications
    # The keys are the name of the modification and the values are dictionaries with kwargs
    modifications_kwargs = {"MLPotential": {"name": "mace-off23-small"}}

    # Define variables that are used in several places to avoid errors
    temperature = 298.15 * unit.kelvin
    dt = 2.0 * unit.femtosecond

    # Set up the mdconfig dictionary for the simulations
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

    # Define the kwargs for the creation of the alchemical states
    # Alternatively, these kwargs can be passed directly to the create_alchemical_states method
    # This uses OpenMM units
    create_alchemical_states_kwargs = {
        "smiles_ligand": "c1ccccc1",
        "smiles_solvent": "[H:2][O:1][H:3]",
        "forcefields": ["openff_unconstrained-2.0.0.offxml", "tip3p.offxml"],
        "temperature": temperature,
        "timestep": dt,  # ignored if integrator is passed
        "pressure": 1.0 * unit.atmospheres,
        "hydrogen_mass": 1.007947 * unit.amu,  # use this for HMR
        "mdconfig_dict": mdconfig_dict,
        "modifications_kwargs": modifications_kwargs,
    }

    # Simulation reoprters
    simulation_reporters = []
    simulation_reporters.append(
        app.StateDataReporter(
            f"stdout_{window}.txt",
            100,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            speed=True,
            volume=True,
            density=True,
        )
    )

    # --------------------------------------------------------------- #
    # Prepare and run the simulations
    # --------------------------------------------------------------- #
    # Create the FES object and add the alchemical states
    fes = FES()
    fes.create_alchemical_states(
        strategy_name="openff",
        lambda_schedule=lambda_schedule,
        **create_alchemical_states_kwargs,
    )

    # Minimize the state of interest
    fes.minimize()

    # Set initial velocities
    fes.set_velocities(temperature=temperature)

    # Equilibrate the state of interest
    fes.equilibrate(10000000)