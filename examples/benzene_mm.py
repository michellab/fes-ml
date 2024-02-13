if __name__ == "__main__":
    import numpy as np
    import openmm as mm
    import openmm.unit as unit

    from fes_ml.fes import FES

    # Set up the alchemical modifications
    n_lambda_q = 3
    n_lambda_lj = 8
    q_windows = np.linspace(1.0, 0.0, n_lambda_q, endpoint=True)
    lj_windows = np.linspace(1.0, 0.0, n_lambda_lj, endpoint=True)

    lambda_schedule = {
        "lambda_q": list(q_windows) + [0.0] * n_lambda_lj,
        "lambda_lj": [1.0] * n_lambda_q + list(lj_windows),
        "lambda_interpolate": [None] * (n_lambda_q) + [None] * (n_lambda_lj),
        "lambda_emle": [None] * (n_lambda_q) + [None] * (n_lambda_lj),
    }

    # Define the dynamics and EMLE parameters
    dynamics_kwargs = {
        "timestep": "1fs",
        "cutoff_type": "pme",
        "cutoff": "12A",
        "integrator": "langevin_middle",
        "temperature": "298.15K",
        "platform": "reference",
        "constraint": "h-bonds",
        "pressure": "1bar",
    }

    emle_kwargs = None

    # Create the FES object to run the simulations
    fes = FES(
        top_file="../data/benzene/benzene_sol.prmtop",
        crd_file="../data/benzene/benzene_sol.inpcrd",
    )
    # Create the alchemical states
    fes.create_alchemical_states(
        alchemical_atoms=list(range(12)),
        lambda_schedule=lambda_schedule,
        dynamics_kwargs=dynamics_kwargs,
        emle_kwargs=emle_kwargs,
    )
    # Minimize the system and equilibrate it during 1 ns
    fes.run_equilibration_batch(1000000, minimize=True)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    fes.run_simulation_batch(1000, 1000)
