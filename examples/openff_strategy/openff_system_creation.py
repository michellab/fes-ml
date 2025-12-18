"""
ML(sol)->ML(gas) free energy calculation for benzene.

This script demonstrates how to calculate a direct absolute hydration free energy at the ML level using MTS integration.
The solute is alchemically modified using a lambda schedule that decouples the solute from the solvent.
At lambda_q=1, the solute-solvent electrostatic interactions are fully turned on.
At lambda_q=0, the solute-solvent electrostatic interactions are fully turned off.
At lambda_lj=1, the solute-solvent van der Waals interactions are fully turned on.
At lambda_lj=0, the solute-solvent van der Waals interactions are fully turned off.

Furthermore, lambda_ml_correction is used at full strength for all states (i.e., lambda_ml_correction=1.0) to introduce a
delta ML correction to the MM energy. This correction is considered part of the slow forces and is integrated twice as
slowly as the fast forces.

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

    if len(sys.argv) == 1:
        raise ValueError("must pass window as positional arguments")

    window = int(sys.argv[1])

    print(f"Window : {window}")

    # --------------------------------------------------------------- #
    # Define the parameters
    # --------------------------------------------------------------- #
    # Set up the alchemical modifications
    # Available modifications are:
    # - ChargeScaling: scale the charges of the solute
    # - LJSoftCore: add a LJ softcore potential to the solute-solvent interactions
    # - MLCorrection: add a delta ML correction to the MM energy
    # - MLInterpolation: interpolate between ML and MM potentials
    # - EMLEPotential: add an EMLE potential to the system
    n_ChargeScaling = 5
    n_LJSoftCore = 11
    q_windows = np.linspace(1.0, 0.0, n_ChargeScaling, endpoint=False)
    lj_windows = np.linspace(1.0, 0.0, n_LJSoftCore)

    lambda_schedule = {
        "ChargeScaling": list(q_windows) + [0.0] * n_LJSoftCore,
        "LJSoftCore": [1.0] * n_ChargeScaling + list(lj_windows),
        "MLCorrection": [1.0] * (n_ChargeScaling + n_LJSoftCore),
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

    # MTS logics
    use_mts = True
    intermediate_steps = 2
    inner_steps = 4
    print("setting up mts")
    if use_mts:
        # Create the MTS class if intermediate steps are defined
        mts = MTS()
        # Multiple time step Langevin integrator
        timestep_groups = [(0, 1), (1, intermediate_steps)]
        if inner_steps:
            timestep_groups.append((2, inner_steps))
        integrator = mts.create_integrator(dt=dt, groups=timestep_groups, temperature=temperature)
    else:
        # The strategy will know how to create the integrator
        integrator = None

    # Define the kwargs for the creation of the alchemical states
    # Alternatively, these kwargs can be passed directly to the create_alchemical_states method
    # This uses OpenMM units
    create_alchemical_states_kwargs = {
        "smiles_ligand": "c1ccccc1",
        "smiles_solvent": "[H:2][O:1][H:3]",
        "integrator": integrator,
        "forcefields": ["openff_unconstrained-2.0.0.offxml", "opc.offxml"],
        "temperature": temperature,
        "timestep": dt,  # ignored if integrator is passed
        "pressure": 1.0 * unit.atmospheres,
        "hydrogen_mass": 1.007947 * unit.amu,  # use this for HMR
        "mdconfig_dict": mdconfig_dict,
        "modifications_kwargs": modifications_kwargs,
    }

    # Simulation parameters
    n_equil_steps = 10000  # 10 ps equilibration
    n_iterations = 3000
    n_steps_per_iter = 1000  # 1 ps per iteration
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
    print("createing fes")
    fes = FES()
    fes.create_alchemical_states(
        strategy_name="openff",
        lambda_schedule=lambda_schedule,
        **create_alchemical_states_kwargs,
    )
    force_group_dict = {"MLCorrection": 0}
    for force in fes.alchemical_states[1].system.getForces():
        if isinstance(force, mm.NonbondedForce):
            if inner_steps:
                force_group_dict[force.getName()] = 1
            else:
                force_group_dict[force.getName()] = 0

    print("setting force ggroups")
    # Set the  force groups for the alchemical states
    if use_mts:
        mts.set_force_groups(
            alchemical_states=fes.alchemical_states,
            force_group_dict=force_group_dict,
            set_reciprocal_space_force_groups=0,
        )

    print("minimising")
    # Minimize the state of interest
    fes.minimize(window=window)

    # Set initial velocities
    fes.set_velocities(temperature=temperature, window=window)

    # Equilibrate the state of interest
    fes.equilibrate(n_equil_steps, window=window)

    # Run single state
    U_kn = fes.run_single_state(
        niterations=n_iterations,
        nsteps=n_steps_per_iter,
        window=int(window),
        reporters=simulation_reporters,
    )

    np.save(f"{window}.npy", np.asarray(U_kn))
