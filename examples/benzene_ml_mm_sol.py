"""
ML(sol)->MM(sol) free energy calculation for benzene in solution using an ML/MM approach (mechanical embedding).

The solute is alchemically modified using a lambda schedule that interpolates between ML and MM potentials.
At lambda_interpolate=1, the solute is fully simulated with an MLP.
At lambda_interpolate=0, the solute is fully simulated with the MM force field.

Authors: Joao Morado
"""

if __name__ == "__main__":
    import numpy as np
    import openmm.app as app

    from fes_ml.fes import FES
    from fes_ml.utils import plot_lambda_schedule

    # Set up the alchemical modifications
    n_lambda_interpolate = 11

    lambda_schedule = {
        "LJSoftCore": [0.5],
        "ChargeScaling": [0.5],
        "MLPotential": [1.0],
    }

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

    # Create the FES object to run the simulations
    fes = FES(
        top_file="../data/benzene/benzene_sage_water.prm7",
        crd_file="../data/benzene/benzene_sage_water.rst7",
    )

    # Create the alchemical states
    print("Creating alchemical states...")
    fes.create_alchemical_states(
        alchemical_atoms=list(range(12)),
        lambda_schedule=lambda_schedule,
        dynamics_kwargs=dynamics_kwargs,
        emle_kwargs=emle_kwargs,
        ml_potential="ani2x",
    )

    # Minimize
    fes.run_minimization_batch(1000)
    # Equilibrate during 1 ns
    fes.run_equilibration_batch(1000000)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(1000, 1000)
    # Save data
    np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
