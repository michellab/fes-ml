"""
MM(sol)->MM(gas) free energy calculation for benzene.

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

    from fes_ml.fes import FES
    from fes_ml.utils import plot_lambda_schedule

    # Set up the alchemical modifications
    n_lambda_q = 3
    n_lambda_lj = 8
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
    U_kln = fes.run_production_batch(1000, 1000)
    np.save("U_kln.npy", np.asarray(U_kln))
