"""
ML(sol)->MM(sol) free energy calculation for benzene in solution using an ML/MM approach (mechanical embedding).

The solute is alchemically modified using a lambda schedule that interpolates between ML and MM potentials.
At MLInterpolation=1, the solute is fully simulated with an MLP.
At MLInterpolation=0, the solute is fully simulated with the MM force field.

Authors: Joao Morado
"""

if __name__ == "__main__":
    import numpy as np
    import openmm as mm
    import openmm.unit as unit

    from fes_ml import FES, MTS
    from fes_ml.utils import plot_lambda_schedule

    # Set up the alchemical modifications
    n_MLInterpolation = 6

    lambda_schedule = {
        "MLInterpolation": np.linspace(1.0, 0.0, n_MLInterpolation),
    }

    plot_lambda_schedule(lambda_schedule, "lambda_schedule_mm_sol_mts.png")

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

    # Create the MTS class
    mts = MTS()
    # Multiple time step Langevin integrator
    # Force group 0, 2 steps (fast forces)
    # Force group 1, 1 step (slow forces)
    integrator = mts.create_integrator(
        dt=1.0 * unit.femtosecond, groups=[(0, 2), (1, 1)]
    )

    # Create the FES object to run the simulations
    fes = FES()

    # Create the alchemical states
    print("Creating alchemical states...")
    fes.create_alchemical_states(
        top_file="../data/benzene/benzene_sage_gas.prm7",
        crd_file="../data/benzene/benzene_sage_gas.rst7",
        alchemical_atoms=list(range(12)),
        lambda_schedule=lambda_schedule,
        dynamics_kwargs=dynamics_kwargs,
        emle_kwargs=emle_kwargs,
        integrator=integrator,
        ml_potential="ani2x",
    )

    # Set the force groups for MTS integration
    mts.set_force_groups(
        alchemical_states=fes.alchemical_states,
        slow_forces=["CustomCVForce"],
        fast_force_group=0,
        slow_force_group=1,
    )

    # Equilibrate during 1 ns
    fes.equilibrate_batch(1000000)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(1000, 1000)
    # Save data
    np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
