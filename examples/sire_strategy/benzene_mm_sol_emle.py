"""
MM(sol, electrostatic) -> MM(gas) free energy calculation for benzene.

This script demonstrates how to calculate the absolute hydration free energy at the MM level
in an electrostatic embedding scheme. The solute is alchemically modified using a lambda
schedule that decouples it from the solvent.

- At EMLEPotential=1, the solute-solvent electrostatic interactions are fully turned on
(electrostatic embedding). At EMLEPotential=0, the solute-solvent electrostatic interactions
are fully turned off (mechanical embedding).
- At ChargeScaling=1, the solute-solvent electrostatic interactions are fully turned on
(mechanical embedding). At ChargeScaling=0, the solute-solvent electrostatic interactions are
fully turned off (mechanical embedding).
- At LJSoftCore=1, the solute-solvent van der Waals interactions are fully turned on. At
LJSoftCore=0, the solute-solvent van der Waals interactions are fully turned off.

Authors: Joao Morado
"""

if __name__ == "__main__":
    import numpy as np

    from fes_ml.fes import FES
    from fes_ml.utils import plot_lambda_schedule

    # Set up the alchemical modifications
    n_EMLEPotential = 6
    n_ChargeScaling = 5
    n_LJSoftCore = 11
    emle_windows = np.linspace(1.0, 0.0, n_EMLEPotential)
    q_windows = np.linspace(1.0, 0.0, n_ChargeScaling, endpoint=False)
    lj_windows = np.linspace(1.0, 0.0, n_LJSoftCore)

    lambda_schedule = {
        "EMLEPotential": list(emle_windows) + [None] * (n_ChargeScaling + n_LJSoftCore),
        "ChargeScaling": [None] * n_EMLEPotential
        + list(q_windows)
        + [0.0] * n_LJSoftCore,
        "LJSoftCore": [None] * n_EMLEPotential
        + [1.0] * n_ChargeScaling
        + list(lj_windows),
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

    emle_kwargs = {"method": "electrostatic", "backend": "sander", "device": "cpu"}

    # Create the FES object to run the simulations
    fes = FES()

    # Create the alchemical states
    fes.create_alchemical_states(
        top_file="../data/benzene/benzene_sage_gas.prm7",
        crd_file="../data/benzene/benzene_sage_gas.rst7",
        alchemical_atoms=list(range(12)),
        lambda_schedule=lambda_schedule,
        dynamics_kwargs=dynamics_kwargs,
        emle_kwargs=emle_kwargs,
    )

    # Minimize
    fes.minimize(1000)
    # Equilibrate during 1 ns
    fes.equilibrate(1000000)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(1000, 1000)
    np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
