"""
ML(gas)->MM(gas) free energy calculation for benzene in vacuum using an ML approach.

This script demonstrates how to perform a free energy calculation for benzene in vacuum using a ML approach.
The solute is alchemically modified using a lambda schedule that interpolates between ML and MM potentials.
At lambda_interpolate=1, the solute is fully simulated with the ML potential (mechanical embedding scheme).
At lambda_interpolate=0, the solute is fully simulated with the FF.

Authors: Joao Morado
"""
if __name__ == "__main__":
    import numpy as np
    import openmm.app as app

    from fes_ml.fes import FES
    from fes_ml.utils import plot_lambda_schedule

    # Set up the alchemical modifications
    n_lambda_interpolate = 11
    interpolate_windows = np.linspace(1.0, 0.0, n_lambda_interpolate)

    lambda_schedule = {"lambda_interpolate": interpolate_windows}

    plot_lambda_schedule(lambda_schedule)

    # Define the dynamics and EMLE parameters
    dynamics_kwargs = {
        "timestep": "1fs",
        "cutoff_type": "NO_CUTOFF",
        "cutoff": "9A",
        "integrator": "langevin_middle",
        "temperature": "298.15K",
        "platform": "cuda",
    }

    emle_kwargs = None

    # Create the FES object to run the simulations
    fes = FES(
        top_file="../data/benzene/benzene_sage_gas.prm7",
        crd_file="../data/benzene/benzene_sage_gas.rst7",
    )

    # Create the alchemical states
    fes.create_alchemical_states(
        alchemical_atoms=list(range(12)),
        lambda_schedule=lambda_schedule,
        dynamics_kwargs=dynamics_kwargs,
        emle_kwargs=emle_kwargs,
        topology=app.AmberPrmtopFile("../data/benzene/benzene_sage_gas.prm7").topology,
        ml_potential="ani2x",
    )

    # Minimize
    fes.run_minimization_batch(1000)
    # Equilibrate during 1 ns
    fes.run_equilibration_batch(1000000)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(1000, 1000)
    # Save data
    np.save("U_kln_ml_mm_gas.npy", np.asarray(U_kln))
