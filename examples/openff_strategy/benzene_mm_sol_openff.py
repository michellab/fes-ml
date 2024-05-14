"""
MM(sol)->MM(gas) free energy calculation for benzene.

This script demonstrates how to calculate a direct absolute hydration free energy at the MM level.
The solute is alchemically modified using a lambda schedule that decouples the solute from the solvent.
At ChargeScaling=1, the solute-solvent electrostatic interactions are fully turned on.
At ChargeScaling=0, the solute-solvent electrostatic interactions are fully turned off.
At LJSoftCore=1, the solute-solvent van der Waals interactions are fully turned on.
At LJSoftCore=0, the solute-solvent van der Waals interactions are fully turned off.

Authors: Joao Morado
"""
if __name__ == "__main__":
    import numpy as np

    from fes_ml.fes import FES

    # Set up the alchemical modifications
    n_ChargeScaling = 5
    n_LJSoftCore = 11
    q_windows = np.linspace(1.0, 0.0, n_ChargeScaling, endpoint=False)
    lj_windows = np.linspace(1.0, 0.0, n_LJSoftCore)

    lambda_schedule = {
        "ChargeScaling": list(q_windows) + [0.0] * n_LJSoftCore,
        "LJSoftCore": [1.0] * n_ChargeScaling + list(lj_windows),
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
    fes = FES()

    # Create the alchemical states
    print("Creating alchemical states...")
    fes.create_alchemical_states(
        strategy_name="openff",
        lambda_schedule=lambda_schedule,
        smarts_ligand="c1ccccc1",
        smarts_solvent="[H:2][O:1][H:3]",
    )

    # Minimize
    fes.run_minimization_batch()
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(300, 300)
    np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
