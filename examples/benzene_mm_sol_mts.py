"""
MM(sol)->MM(gas) free energy calculation for benzene using a multiple time step Langevin integrator

This script demonstrates how to calculate a direct absolute hydration free energy at the MM level.
The solute is alchemically modified using a lambda schedule that decouples the solute from the solvent.
At lambda_q=1, the solute-solvent electrostatic interactions are fully turned on.
At lambda_q=0, the solute-solvent electrostatic interactions are fully turned off.
At lambda_lj=1, the solute-solvent van der Waals interactions are fully turned on.
At lambda_lj=0, the solute-solvent van der Waals interactions are fully turned off.

Authors: Joao Morado
"""

if __name__ == "__main__":
    import numpy as np
    import openmm as mm
    import openmm.unit as unit

    from fes_ml.fes import FES
    from fes_ml.utils import plot_lambda_schedule

    # Set up the alchemical modifications
    n_lambda_q = 5
    n_lambda_lj = 11
    q_windows = np.linspace(1.0, 0.0, n_lambda_q, endpoint=False)
    lj_windows = np.linspace(1.0, 0.0, n_lambda_lj)

    lambda_schedule = {
        "lambda_q": list(q_windows) + [0.0] * n_lambda_lj,
        "lambda_lj": [1.0] * n_lambda_q + list(lj_windows),
    }

    plot_lambda_schedule(lambda_schedule)

    # Define the dynamics and EMLE parameters
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

    emle_kwargs = None

    # Multiple time step Langevin integrator
    # Force group 0, 2 steps (fast forces)
    # Force group 1, 1 step (slow forces)
    groups = [(0, 2), (1, 1)]
    integrator = mm.MTSLangevinIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 1 * unit.femtosecond, groups
    )

    # Create the FES object to run the simulations
    fes = FES(
        top_file="../data/benzene/benzene_sage_water.prm7",
        crd_file="../data/benzene/benzene_sage_water.rst7",
    )

    # Create the alchemical states
    fes.create_alchemical_states(
        alchemical_atoms=list(range(12)),
        lambda_schedule=lambda_schedule,
        dynamics_kwargs=dynamics_kwargs,
        emle_kwargs=emle_kwargs,
        integrator=integrator,
    )

    # Set the force groups
    fes.set_force_groups(
        slow_forces=["NonbondedForce", "CustomNonbondedForce"],
        fast_force_group=0,
        slow_force_group=1,
    )

    # Equilibrate during 1 ns
    fes.run_equilibration_batch(1000000)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(1000, 1000)
    np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
