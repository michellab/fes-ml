import logging
import math
import os
import sys
from argparse import ArgumentParser
from typing import Optional, Union

import numpy as np
import openff.units as offunit
import openmm as mm
import openmm.app as app
import openmm.unit as unit

os.environ["FES_ML_FILTER_LOGGERS"] = "0"

from fes_ml import FES, MTS

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

def main(args):
    # arguments
    smiles_ligand = args.smiles_ligand  # "c1ccccc1"
    folder = args.folder
    windows = int(args.windows)
    timestep = float(args.timestep)
    use_mts = args.use_mts
    intermediate_steps = int(args.intermediate_steps)
    inner_steps = int(args.inner_steps)
    use_hmr = args.use_hmr
    hmr_factor = float(args.hmr_factor)
    dont_split_nonbonded = args.dont_split_nonbonded
    lig_ff = args.ligand_forcefield
    sampling_time = args.sampling
    vacuum_simulation = args.vacuum_simulation
    vanish_ligand = bool(args.vanish_ligand)
    ml_application = args.ml_application.lower()
    run_window = int(args.run_window)

    logger.info(f"using the following arguments: {args}")

    try:
        os.makedirs(folder)
    except:
        pass

    # --------------------------------------------------------------- #
    # Define the parameters
    # --------------------------------------------------------------- #
    # Set up the alchemical modifications
    # Available modifications are:
    # - ChargeScaling: scale the charges of the solute
    # - LJSoftCore: add a LJ softcore potential to the solute-solvent interactions
    # - MLCorrection: add a delta ML correction to the MM energy
    # - MLInterpolation: interpolate between ML and MM potentials
    # - EMLEPotential: add an EMLE potential to the system

    # split the windows
    n_ChargeScaling = math.ceil(windows * 0.3)
    n_LJSoftCore = math.floor(windows * 0.7)
    assert (n_ChargeScaling + n_LJSoftCore) == windows
    q_windows = np.linspace(1.0, 0.0, n_ChargeScaling, endpoint=False)
    lj_windows = np.linspace(1.0, 0.0, n_LJSoftCore)

    if vanish_ligand:
        lambda_schedule = {
            "ChargeScaling": list(q_windows) + [0.0] * n_LJSoftCore,
            "LJSoftCore": [1.0] * n_ChargeScaling + list(lj_windows),
        }
    else:
        lambda_schedule = {}
    
    if ml_application == "correction" and vanish_ligand == True:
        logger.info("Fully applying the ML correction at each window as the ligand is also vanishing...")
        lambda_schedule["MLCorrection"] = [1.0] * (n_ChargeScaling + n_LJSoftCore)
    elif ml_application == "correction" and not vanish_ligand:
        logger.info("Applying the ML correction progressively with each at each window...")
        lambda_schedule["MLCorrection"] = np.linspace(0.0, 1.0, windows)
    elif ml_application == "interpolation" and vanish_ligand == True:
        raise ValueError("ML interpolation is not compatible with vanishing the ligand simulataneously.")
    elif ml_application == "interpolation" and not vanish_ligand:   
        logger.info("Applying the ML interpolation progressively with each at each window...")
        lambda_schedule["MLInterpolation"] = np.linspace(0.0, 1.0, windows)
        
    else:
        raise ValueError("ml_application must be 'correction' or 'interpolation'.")

    logger.info(f"the lambda schedule is: {lambda_schedule}")

    # Modifications kwargs
    # This dictionary is used to pass additional kwargs to the modifications
    # The keys are the name of the modification and the values are dictionaries with kwargs
    if lig_ff.lower() == "mace":
        modifications_kwargs = {"MLPotential": {"name": "mace-off23-small"}}
    elif lig_ff.lower() == "ani2x":
        modifications_kwargs = {"MLPotential": {"name": "ani2x"}}
    elif lig_ff.upper() == "NONE":
        modifications_kwargs = {}
        logger.info("not using any ML potential...")
    else:
        raise ValueError("ligand forcefield must be mace or ani2x .")

    # Define variables that are used in several places to avoid errors
    temperature = 298.15 * unit.kelvin
    dt = timestep * unit.femtosecond

    # Set up the mdconfig dictionary for the simulations
    # This is the default mdconfig dictionary, meaning that if this dictionary
    # is not passed to the FES object, these are the values that will be used.
    if not vacuum_simulation:
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
    elif vacuum_simulation:
        mdconfig_dict = {
            "periodic": False,
            "constraints": "h-bonds",
            "vdw_method": "no-cutoff",
            "mixing_rule": "lorentz-berthelot",
            "switching_function": True,
            "coul_method": "no-cutoff",
        }
    else:
        raise ValueError("vacuum simulation must be True or False.")

    # MTS logics
    if use_mts:
        logger.info("setting up mts")
        # Create the MTS class if intermediate steps are defined
        mts = MTS()
        # Multiple time step Langevin integrator
        # can always add all force groups, as unassigned are later just assigned to the fastest group
        timestep_groups = [(0, 1), (2, inner_steps)]
        if intermediate_steps:
            timestep_groups.append((1, intermediate_steps))
        integrator = mts.create_integrator(
            dt=dt, groups=timestep_groups, temperature=temperature
        )
    else:
        logger.info("no mts setup...")
        # The strategy will know how to create the integrator
        integrator = None

    if use_hmr:
        hydrogen_mass = 1.007947 * hmr_factor * unit.amu
    else:
        hydrogen_mass = 1.007947 * unit.amu

    # Define the kwargs for the creation of the alchemical states
    # Alternatively, these kwargs can be passed directly to the create_alchemical_states method
    # This uses OpenMM units
    if not vacuum_simulation:
        create_alchemical_states_kwargs = {
            "smiles_ligand": smiles_ligand,
            "smiles_solvent": "[H:2][O:1][H:3]",
            "integrator": integrator,
            "forcefields": [
                "openff_unconstrained-2.0.0.offxml", # "gaff"
                "opc.offxml",  # "amber14/tip3p.xml",
            ],  # tip3p.offxml , opc.offxml
            "temperature": temperature,
            "timestep": dt,  # ignored if integrator is passed
            "pressure": 1.0 * unit.atmospheres,
            "hydrogen_mass": hydrogen_mass,
            "mdconfig_dict": mdconfig_dict,
            "modifications_kwargs": modifications_kwargs,
            "write_pdb": True,
            "write_gro": True,
            "tmp_dir": f"{folder}/tmp",
            "write_system_xml": True
        }
    elif vacuum_simulation:
        create_alchemical_states_kwargs = {
            "smiles_ligand": smiles_ligand,
            "smiles_solvent": None,
            "integrator": integrator,
            "forcefields": [
                "openff_unconstrained-2.0.0.offxml"
            ],
            "temperature": temperature,
            "timestep": dt,  # ignored if integrator is passed
            "pressure": 1.0 * unit.atmospheres,
            "hydrogen_mass": hydrogen_mass,
            "mdconfig_dict": mdconfig_dict,
            "modifications_kwargs": modifications_kwargs,
            "write_pdb": True,
            "write_gro": False,
            "tmp_dir": f"{folder}/tmp",
            "write_system_xml": True
        }

    # --------------------------------------------------------------- #
    # Prepare and run the simulations
    # --------------------------------------------------------------- #
    # Create the FES object and add the alchemical states
    logger.info("createing fes")
    logger.info(f"using the following kwargs: {create_alchemical_states_kwargs}")
    fes = FES(output_prefix=f"{folder}/fes")
    fes.create_alchemical_states(
        strategy_name="openff",
        lambda_schedule=lambda_schedule,
        **create_alchemical_states_kwargs,
    )

    # Set the  force groups for the alchemical states
    if use_mts:
        logger.info("setting force groups")

        if ml_application == "correction":
            force_group_dict = {"MLCorrection": 0}
        elif ml_application == "interpolation":
            force_group_dict = {"MLInterpolation": 0}
        
        for force in fes.alchemical_states[1].system.getForces():
            if isinstance(force, mm.NonbondedForce):
                if intermediate_steps:
                    force_group_dict[force.getName()] = 1
                else:
                    force_group_dict[force.getName()] = 0
        # by default, the rest of the forces are set to the fastest group

        if dont_split_nonbonded:
            logger.info("not splitting the non bonded interactions...")
            reciprocal_space_force_group = None
        else:
            logger.info("splitting the non bonded interactions if theres intermediate steps...")
            reciprocal_space_force_group = 0

        logger.info(f"using the follwing force group dictionary: {force_group_dict}")
        mts.set_force_groups(
            alchemical_states=fes.alchemical_states,
            force_group_dict=force_group_dict,
            set_reciprocal_space_force_groups=reciprocal_space_force_group,
        )

    # run for only the single specified window
    if run_window < 0 or run_window >= windows:
        raise ValueError(f"run_window must be between 0 and {windows - 1}, as there are {windows} windows.")
        exit()

    logger.info(f"Window : {run_window}")

    # Simulation parameters
    n_equil_steps = int(10000 / timestep)  # 10 ps equilibration
    n_steps_per_iter = int(1000 / timestep) # 1 ps per iteration, as this is the freq the energy is evaluated at
    total_steps = int((1000000 / timestep) * sampling_time) # steps per ns * the sampling time
    n_iterations = int(total_steps / n_steps_per_iter)

    try:
        assert n_iterations ==  ((1000000 / timestep) * sampling_time) / (1000 / timestep)
    except AssertionError as e:
        logger.error(e)
        logger.error(f"n_iterations is not the same as the values if int is not applied")

    simulation_reporters = []
    simulation_reporters.append(
        app.StateDataReporter(
            f"{folder}/stdout_{run_window}.txt",
            100,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            speed=True,
            volume=True,
            density=True,
        )
    )
    simulation_reporters.append(app.DCDReporter(
        f"{folder}/dcd_{run_window}.dcd", n_steps_per_iter))

    logger.info("minimising...")
    # Minimize the state of interest
    fes.minimize(window=run_window)

    # Set initial velocities
    fes.set_velocities(temperature=temperature, window=run_window)

    # Equilibrate the state of interest
    logger.info("equilibrating...")
    fes.equilibrate(n_equil_steps, window=run_window)

    # Run single state
    logger.info("running the single state...")
    U_kn = fes.run_single_state(
        niterations=n_iterations,
        nsteps=n_steps_per_iter,
        window=int(run_window),
        reporters=simulation_reporters,
    )

    np.save(f"{folder}/{run_window}.npy", np.asarray(U_kn))
    logger.info(f"saved to: {folder}/{run_window}.npy")


if __name__ == "__main__":
    # accept all options as arguments
    parser = ArgumentParser(description="run the AFE with ML correction")
    parser.add_argument(
        "-s",
        "--smiles",
        dest="smiles_ligand",
        type=str,
        default=None,
        required=True,
        help="smiles of the ligand",
    )
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        type=str,
        default=None,
        required=True,
        help="folder path for the run",
    )
    parser.add_argument(
        "-w",
        "--windows",
        dest="windows",
        type=int,
        default=11,
        help="Number of lambda windows",
    )
    parser.add_argument(
        "-dt",
        "--timestep",
        dest="timestep",
        type=float,
        default=2,
        help="Timestep in fs",
    )
    parser.add_argument(
        "-st",
        "--sampling",
        dest="sampling",
        type=float,
        default=0.01,
        help="runtime in ns. Default is 0.01 ns, which is 10 ps.",
    )
    parser.add_argument(
        "--use-mts",
        dest="use_mts",
        action="store_true",
        help="Whether to use MTS for the run. This will then use the inner and intermediate step arguments",
    )
    parser.add_argument(
        "-in",
        "--inner-steps",
        dest="inner_steps",
        type=int,
        default=4,
        help="Number of inner steps",
    )
    parser.add_argument(
        "-int",
        "--intermediate-steps",
        dest="intermediate_steps",
        type=int,
        default=0,
        help="Number of intermediate steps. Put as 0 to not use.",
    )
    parser.add_argument(
        "--ligand-forcefield",
        dest="ligand_forcefield",
        type=str,
        default="mace",
        choices=["mace", "ani2x", "None"],
        help="The ligand forcefield to use. None will use the default OpenFF forcefield and no ML will be applied.",
    )
    parser.add_argument(
        "--use-hmr",
        dest="use_hmr",
        action="store_true",
        help="Whether to use HMR for the runs",
    )
    parser.add_argument(
        "--hmr-factor",
        dest="hmr_factor",
        type=float,
        default=3,
        help="Which HMR factor to use",
    )
    parser.add_argument(
        "--ml-application",
        dest="ml_application",
        type=str,
        default="correction",
        help="Whether to use ML correction or interpolation. ",
    )
    parser.add_argument(
        "--do-not-vanish-ligand",
        dest="vanish_ligand",
        action="store_false",
        help="Whether to vanish the ligand or not. "
        "Default is vanishing the ligand (no flag), if the flag is present the ligand will not be alchemically perturbed.",
    )
    parser.add_argument(
        "--do-not-split-nonbonded",
        dest="dont_split_nonbonded",
        action="store_true",
        help="Whether to not split the non-bonded interactions"
        "Default the reciprocal space force group is the slowest.",
    )
    parser.add_argument(
        "--vacuum",
        dest="vacuum_simulation",
        action="store_true",
        help="Whether it is a vacuum simulation or not. "
        "Default is a solvent simulation. "
    )
    parser.add_argument(
        "-rw",
        "--run_window",
        dest="run_window",
        type=int,
        default=0,
        help="The window to run the simulation at.",
    )
    args = parser.parse_args()

    main(args)
