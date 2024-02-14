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

    from fes_ml.fes import FES
    from fes_ml.utils import plot_lambda_schedule

    # Set up the alchemical modifications
    n_lambda_emle = 10
    emle_windows = np.linspace(1.0, 0.0, n_lambda_emle)

    lambda_schedule = {"lambda_emle": emle_windows}

    plot_lambda_schedule(lambda_schedule)

    # Define the dynamics and EMLE parameters
    dynamics_kwargs = {
        "timestep": "1fs",
        "cutoff_type": "pme",
        "cutoff": "12A",
        "integrator": "langevin_middle",
        "temperature": "298.15K",
        "platform": "reference",
        "constraint": "h-bonds",
    }

    emle_kwargs = None

    # Create the FES object to run the simulations
    fes = FES(
        top_file="/Users/admin/workspace/git_repos/fes-ml/data/benzene/benzene_sol.prmtop",
        crd_file="/Users/admin/workspace/git_repos/fes-ml/data/benzene/benzene_sol.inpcrd",
    )

    # Create the alchemical states
    fes.create_alchemical_states(
        alchemical_atoms=list(range(12)),
        lambda_schedule=lambda_schedule,
        dynamics_kwargs=dynamics_kwargs,
        emle_kwargs=emle_kwargs,
    )

    # Minimize the system and equilibrate it during 1 ns
    fes.run_equilibration_batch(1000000, minimize=False)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    fes.run_simulation_batch(1000, 1000)
