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
    import openff.units as offunit
    import openmm.unit as unit

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

    # Set up the mdconfig dictionary for the simulations
    # This is the default mdconfig dictionary, meaning that if this dictionary
    # is not passed to the FES object, these values will be used.
    mdconfig_dict = {
        "periodic": True,
        "constraints": "h-bonds",
        "vdw_method": "cutoff",
        "vdw_cutoff": offunit.Quantity(12.0, "angstrom"),
        "mixing_rule": "lorentz-berthelot",
        "switching_function": True,
        "switching_distance": offunit.Quantity(11.0, "angstrom"),
        "coul_method": "pme",
        "coul_cutoff": offunit.Quantity(12.0, "angstrom"),
    }

    # Create the FES object to run the simulations
    fes = FES()

    # Create the alchemical states
    fes.create_alchemical_states(
        strategy_name="openff",
        lambda_schedule=lambda_schedule,
        smiles_ligand="c1ccccc1",
        smiles_solvent="[H:2][O:1][H:3]",
        integrator=None,  # None means that the default integrator will be used (LangevinMiddleIntegrator as the temperature is set to 298.15 K)
        forcefields=["openff-2.0.0.offxml", "tip3p.offxml"],
        temperature=298.15 * unit.kelvin,
        timestep=1.0 * unit.femtosecond,
        pressure=1.0 * unit.atmospheres,
        hydrogen_mass=1.007947 * unit.amu,
        mdconfig_dict=mdconfig_dict,
    )

    # Minimize the batch of states
    fes.minimize()
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(1000, 1000)
    np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
