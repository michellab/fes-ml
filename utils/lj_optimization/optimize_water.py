"""Script to optimize the LJ parameters of the TIP3P water model."""

if __name__ == "__main__":
    import logging
    import os

    import numpy as np
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    import pandas as pd
    from lj_optimizer import LJOptimizer
    from openmmml import MLPotential

    from fes_ml.fes import FES

    logger = logging.getLogger()
    # ------------------------------------------------------------------------------------------
    # SET UP THE ALCHEMICAL SYSTEM
    # ------------------------------------------------------------------------------------------
    logger.info("Setting up the alchemical system")
    # Set up the alchemical modifications
    lambda_schedule = {
        "ChargeScaling": [1.0],
        "LJSoftCore": [1.0],
        "EMLEPotential": [1.0],
    }

    # Define the dynamics and EMLE parameters
    dynamics_kwargs = {
        "timestep": "1fs",
        "cutoff_type": "NO_CUTOFF",
        "cutoff": "999A",
        "constraint": "none",
        "integrator": "langevin",
        "temperature": "298.15K",
        "platform": "Reference",
        "map": {"use_dispersion_correction": False, "tolerance": 0.0005},
    }

    emle_kwargs = {
        "method": "electrostatic",
        "device": "cpu",
        "backend": "torchani",
    }

    # Create the FES object to run the simulations
    top_file = "data/benzene_water_dimer.prm7"
    crd_file = "data/benzene_water_dimer.rst7"

    # Create the FES object
    fes = FES(
        top_file=top_file,
        crd_file=crd_file,
    )

    benzene_atoms = list(range(3, 3 + 12))
    logger.info(f"Alchemical atoms: {benzene_atoms}")

    # Create one alchemical state
    fes.create_alchemical_states(
        alchemical_atoms=benzene_atoms,
        lambda_schedule=lambda_schedule,
        dynamics_kwargs=dynamics_kwargs,
        emle_kwargs=emle_kwargs,
        minimise_iterations=0,
    )

    # ------------------------------------------------------------------------------------------
    # CREATE ML SYSTEM FOR BENZENE
    # ------------------------------------------------------------------------------------------
    logger.info("Creating the ML system for benzene")
    alchemical_system_prmtop = os.path.join(
        os.getcwd(), "tmp_fes_ml_sire/alchemical_subsystem.prm7"
    )
    topology_ml = app.AmberPrmtopFile(alchemical_system_prmtop).topology
    potential_ml = MLPotential("ani2x")
    system_ml = potential_ml.createSystem(topology_ml)
    context_ml = mm.Context(
        system_ml,
        mm.LangevinIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond
        ),
    )

    # ------------------------------------------------------------------------------------------
    # LOAD THE DESS66x8 DATA FOR THE BENZENE-WATER DIMER
    # ------------------------------------------------------------------------------------------
    logger.info("Loading the DESS66x8 dataset")
    # Load the dataset
    dataset_path = "/Users/admin/workspace/dataset/DESS66x8.csv"
    group_id = "3057170"
    df = pd.read_csv(dataset_path)
    df = df[df["group_id"] == int(group_id)]

    # Load the configurations and QM energies from the
    natoms0 = df["natoms0"].values[0]
    natoms1 = df["natoms1"].values[1]

    xyz = df["xyz"].values
    xyz = np.array([np.array([float(j) for j in i.split(" ")]) for i in xyz])

    # Extract the configurations into a 3xN_atomsxN_configs array
    configurations = np.zeros((3, natoms0 + natoms1, xyz.shape[0]))

    for i in range(xyz.shape[0]):
        for j in range(0, xyz.shape[1], 3):
            configurations[:, j // 3, i] = xyz[i, j : j + 3]

    configurations = configurations * 0.1  # Convert from Angstrom to nm

    # Extract the energy values
    energies = df["cbs_CCSD(T)_all"].values
    energies = energies * 4.184  # Convert from Hartree to kcal/mol

    # ------------------------------------------------------------------------------------------
    # PERFORM THE LJ OPTIMIZATION
    # ------------------------------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("Energy decomposition for the initial system (kJ/mol)")
    alc_state = fes.alchemical_states[0]

    logger.info("Performing the LJ optimization")
    lj_optimizer = LJOptimizer(
        context=alc_state.context,
        topology=alc_state.topology,
        configurations=configurations,
        energy_qm=energies,
        verbose=False,
    )

    # Compute the MM energy before the optimization
    mm_energy = lj_optimizer.compute_energy(configurations)

    # Calculate the offset energy
    configurations_ml = configurations[:, benzene_atoms, :]
    energy_offset = lj_optimizer.compute_energy(configurations_ml, context_ml)
    lj_optimizer._energy_offset = energy_offset

    # Remove the harmonic bond force for the Water molecule
    for force in alc_state.system.getForces():
        if force.getName() == "HarmonicBondForce":
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                if p1 not in benzene_atoms and p2 not in benzene_atoms:
                    force.setBondParameters(i, p1, p2, length, 0.0)

            force.updateParametersInContext(alc_state.context)

    # Define the dictionary containing the residues and the atoms to optimize
    # Here we only optimize the LJ parameters of the oxygen atoms in the water molecule
    res_atoms = {
        "HOH": ["O"],
    }

    lj_optimizer.run_optimization(
        residues_atoms=res_atoms,
        force_type=mm.CustomNonbondedForce,
        method="trust-constr",
        options={"verbose": 3, "maxiter": 1000, "gtol": 1e-6, "disp": True},
    )

    lj_optimizer.write_optimized_parameters("optimized_parameters.txt")

    # ------------------------------------------------------------------------------------------
    # PLOT THE RESULTS
    # ------------------------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid", font_scale=1.5, palette="Set1")

    fig, axs = plt.subplots(1, 1, figsize=(18, 7))
    plt.title("Benzene (ANI-2x) - Water (TIP3P) dimer")
    plt.plot(
        (mm_energy - lj_optimizer._energy_offset) / 4.184, label="Original", marker="o"
    )
    plt.plot(
        (lj_optimizer.compute_energy(configurations) - lj_optimizer._energy_offset)
        / 4.184,
        label="Optimized",
        marker="o",
    )
    plt.plot(
        energies / 4.184,
        label="CCSD(T)/CBS",
        marker="o",
        linestyle="--",
        color="black",
    )
    plt.ylabel("Energy / kcal mol$^{-1}$")
    plt.xlabel("Configuration")
    plt.legend()
    plt.show()

    lj_optimizer.update_system_parameters_from_file("optimized_parameters.txt")
