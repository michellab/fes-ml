"""
MM(sol, electrostatic) -> MM(gas) free energy calculation for benzene.

This script demonstrates how to calculate the absolute hydration free energy at the MM level in an electrostatic embedding scheme.
The solute is alchemically modified using a lambda schedule that decouples it from the solvent.

At EMLEPotential=1, the solute-solvent electrostatic interactions are fully turned on (electrostatic embedding).
At EMLEPotential=0, the solute-solvent electrostatic interactions are fully turned off (mechanical embedding).
At ChargeScaling=1, the solute-solvent electrostatic interactions are fully turned on (mechanical embedding).
At ChargeScaling=0, the solute-solvent electrostatic interactions are fully turned off (mechanical embedding).
At LJSoftCore=1, the solute-solvent van der Waals interactions are fully turned on.
At LJSoftCore=0, the solute-solvent van der Waals interactions are fully turned off.

Authors: Joao Morado
"""
if __name__ == "__main__":
    import numpy as np

    from fes_ml.fes import FES

    # Set up the alchemical modifications
    n_EMLEPotential = 6
    n_ChargeScaling = 5
    n_LJSoftCore = 11
    emle_windows = np.linspace(1.0, 0.0, n_EMLEPotential)
    q_windows = np.linspace(1.0, 0.0, n_ChargeScaling, endpoint=False)
    lj_windows = np.linspace(1.0, 0.0, n_LJSoftCore)

    lambda_schedule = {
        "EMLEPotential": [1.0, 0.0],
    }

    # Define the dynamics and EMLE parameters
    dynamics_kwargs = {
        "timestep": "1fs",
        "cutoff_type": "PME",
        "cutoff": "12A",
        "constraint": "h_bonds",
        "integrator": "langevin_middle",
        "temperature": "298.15K",
        "platform": "cuda",
        "map": {"use_dispersion_correction": True, "tolerance": 0.0005},
    }

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
    )

    # Minimize
    fes.run_minimization_batch(1000)
    # Equilibrate during 1 ns
    fes.run_equilibration_batch(1000000)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(1000, 1000)
    np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
