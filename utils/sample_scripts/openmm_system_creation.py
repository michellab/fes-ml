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
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    from fes_ml.fes import FES
    from fes_ml.utils import plot_lambda_schedule

    if len(sys.argv) != 3:
        raise ValueError("must pass script and window as positional arguments")

    script = sys.argv[1]
    window = sys.argv[2]

    print(f"Script : {script}")
    print(f"Window : {window}")

    sdf_file = ""  # path to sdf file

    # Set up the alchemical modifications
    n_lambda_interpolate = 6

    lambda_schedule = {}

    # this is a diff no of windows than w the batch scripts
    # Set up the alchemical modifications
    # n_lambda_q = 5
    # n_lambda_lj = 11tcctctc
    # q_windows = np.linspace(1.0, 0.0, n_lambda_q, endpoint=False)
    # lj_windows = np.linspace(1.0, 0.0, n_lambda_lj)

    lambda_schedule = {
        "lambda_interpolate": np.linspace(1.0, 0.0, n_lambda_interpolate),
        "lambda_ml_correction": [1.0] * (n_lambda_interpolate),
    }

    # Define the dynamics and EMLE parameters
    dynamics_kwargs = {
        "timestep": "1fs",
        "cutoff_type": "PME",
        "cutoff": "12A",
        "integrator": "langevin_middle",
        "temperature": "298.15K",
        "pressure": "1atm",
        "platform": "cuda",
        "map": {"use_dispersion_correction": True, "tolerance": 0.0005},
    }

    create_system_kwargs = {
        "ligand_forcefield": "openff",
        "water_model": "OPC",
        "HMR": True,
    }

    temperature = float(dynamics_kwargs["temperature"][:-1])
    dt = float(dynamics_kwargs["timestep"][:-1])

    innersteps = 2
    innerinnersteps = 4

    if dynamics_kwargs["integrator"] == "langevin_middle":
        # 1. normal Langevin Middle. Just choose timestep:
        integrator = mm.LangevinMiddleIntegrator(
            temperature * unit.kelvin, 1 / unit.picosecond, dt * unit.femtosecond
        )

    elif dynamics_kwargs["integrator"] == "MTS":
        # 2. Multiple timestep langevin middle. Choose outer timestep and number of inner steps.
        # group 0 is slow forces, group 1 is fast forces, group 2 is fastest forces
        timestep_groups = [(0, 1), (1, innersteps)]
        if innerinnersteps:
            timestep_groups.append((2, innerinnersteps))
        integrator = mm.MTSLangevinIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picosecond,
            dt * unit.femtosecond,
            timestep_groups,
        )

    # Create the FES object to run the simulations
    fes = FES(
        sdf_file=sdf_file,
    )

    # Create the alchemical states
    print("Creating alchemical states...")
    fes.create_alchemical_states(
        strategy_name="openmm",
        lambda_schedule=lambda_schedule,
        dynamics_kwargs=dynamics_kwargs,
        integrator=integrator,
        ml_potential="mace",
        create_system_kwargs=create_system_kwargs**dynamics_kwargs,
    )

    # TODO how get the created system?
    # TODO choose force groups from system so can set using set_force_groups

    force_group_dict = {}

    """
    how the forces should set
    print("system forces:")
    for i, force in enumerate(system.getForces()):

        if isinstance(force, (mm.CustomCVForce)):
            group = 'slow'
        elif isinstance(force, mm.NonbondedForce):
            if innerinnersteps:
                group = 'fast'
            else:
                group = 'slow'
            force.setReciprocalSpaceForceGroup(0)
        else:
            if innerinnersteps:
                group = 'fastest'
            else:
                group = 'fast'

        print(i, force, group)
        force.setForceGroup( {'fastest': 2, 'fast': 1, 'slow': 0}[group])
        if isinstance(force, mm.NonbondedForce):
            force.setReciprocalSpaceForceGroup(0)

    """

    # Set the force groups
    fes.set_force_groups(force_group_dict=force_group_dict)

    # # Equilibrate during 1 ns
    # fes.equilibrate_batch(10000) # 1000000
    # # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    # U_kln = fes.run_production_batch( 10, 1000, # 1000, 1000, # niterations, nsteps
    #                                  reporters=app.StateDataReporter(step=True, # f'{folderpath}/stdout_eq_{w}.txt', n_steps,
    #                 potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True,
    #                 speed=True, volume=True, density=True))

    # # Save data
    # np.save("U_kln_mm_sol.npy", np.asarray(U_kln))

    # Minimize
    fes.minimize_batch(1000)
    # Run single state
    U_kn = fes.run_single_state(
        1000,
        1000,
        window=int(window),
        reporters=[
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
        ],
    )
    np.save(f"{script}_{window}.npy", np.asarray(U_kn))
