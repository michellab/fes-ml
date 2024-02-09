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

    lambda_dict = {
        "lambda_q": list(q_windows) + [0.0] * (n_lambda_lj - 1),
        "lambda_lj": [1.0] * (n_lambda_q - 1) + list(lj_windows),
        "lambda_interpolate": [None] * (n_lambda_q) + [None] * (n_lambda_lj),
    }

    # Create the FES object to run the simulations
    fes = FES(
        topology_format="AMBER",
        top_file="../data/benzene/benzene_sol.prmtop",
        crd_format="AMBER",
        crd_file="../data/benzene/benzene_sol.inpcrd",
        platform_name="CUDA",
    )
    fes.create_alchemical_states(lambda_dict, alchemical_atoms=list(range(12)))
    for alc in fes._alchemical_states:
        alc.system.addForce(mm.MonteCarloBarostat(1.0*unit.bar, 300*unit.kelvin))

    fes.create_simulation_batch()
    # Minimize the system and equilibrate it during 1 ns
    fes.run_equilibration_batch(1000000, minimize=True)
    # Sample 1000 times every ps (i.e. 1 ns of simulation per state)
    fes.run_simulation_batch(1000, 1000)
