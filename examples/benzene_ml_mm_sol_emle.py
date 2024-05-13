"""
ML(sol, electrostatic) -> MM(sol, mechanical) free energy calculation for benzene.

This script demonstrates how to calculate the free energy difference between a solvated
MLP/MM system with electrostatic embedding and a solvated MM system with mechanical embedding.

The electostatic embedding is added through the EMLEPotential modification. Because here
we want the EMLEPotential to vary along MLInterpolation, we add it as a post-dependency of the
MLInterpolation modification.

Authors: Joao Morado
"""
if __name__ == "__main__":
    import numpy as np
    import openmm as mm

    from fes_ml.fes import FES
    from fes_ml.alchemical.modifications.ml_interpolation import MLInterpolationModification

    # Set up the alchemical modifications
    n_interpolation = 3

    lambda_schedule = {
        "MLInterpolation" : np.linspace(1.0, 0.0, n_interpolation),
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

    # Add the EMLEPotential as a post-dependency of the MLInterpolation
    # This will apply the EMLEPotential modification after the MLInterpolation.
    MLInterpolationModification.add_post_dependency("EMLEPotential")

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
        minimise_iterations=0,
    )

    # Minimize
    fes.run_minimization_batch(1000)
    # Equilibrate during 1 ns
    fes.run_equilibration_batch(1000000)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    U_kln = fes.run_production_batch(1000, 1000)
    np.save("U_kln_mm_sol.npy", np.asarray(U_kln))
