"""
ML(sol)->ML(gas) free energy calculation for benzene.

This script demonstrates how to calculate a direct absolute hydration free energy at the ML level using MTS integration.
The solute is alchemically modified using a lambda schedule that decouples the solute from the solvent.
At ChargeScaling=1, the solute-solvent electrostatic interactions are fully turned on.
At ChargeScaling=0, the solute-solvent electrostatic interactions are fully turned off.
At LJSoftCore=1, the solute-solvent van der Waals interactions are fully turned on.
At LJSoftCore=0, the solute-solvent van der Waals interactions are fully turned off.

Furthermore, MLCorrection is used at full strength for all states (i.e., MLCorrection=1.0) to introduce a
delta ML correction to the MM energy. This correction is considered part of the slow forces and is integrated twice as
slowly as the fast forces.

Authors: Joao Morado
"""

if __name__ == "__main__":
    import numpy as np
    import openmm.unit as unit

    from fes_ml import FES, MTS

    # Set up the alchemical modifications
    n_ChargeScaling = 5
    n_LJSoftCore = 11
    q_windows = np.linspace(1.0, 0.0, n_ChargeScaling, endpoint=False)
    lj_windows = np.linspace(1.0, 0.0, n_LJSoftCore)

    lambda_schedule = {
        "ChargeScaling": list(q_windows) + [0.0] * n_LJSoftCore,
        "LJSoftCore": [1.0] * n_ChargeScaling + list(lj_windows),
        "MLCorrection": [1.0] * (n_ChargeScaling + n_LJSoftCore),
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
        "platform": "reference",
        "map": {"use_dispersion_correction": True, "tolerance": 0.0005},
    }

    emle_kwargs = None

    # Create the MTS class
    mts = MTS()
    # Multiple time step Langevin integrator
    # Force group 0, 2 steps (fast forces)
    # Force group 1, 1 step (slow forces)
    integrator = mts.create_integrator(dt=1.0 * unit.femtosecond, groups=[(0, 2), (1, 1)])

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
    fes.equilibrate(1000000)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(1000, 1000)
    # Save data
    np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
