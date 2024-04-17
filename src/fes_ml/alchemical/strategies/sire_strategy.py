"""Sire alchemical state creation strategy."""

import logging
import shutil as _shutil
from copy import deepcopy as _deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as _np
import openmm as _mm
import openmm.app as _app
import sire as _sr
from emle.calculator import EMLECalculator as _EMLECalculator

from ...utils import energy_decomposition as energy_decomposition
from ..alchemical_state import AlchemicalState
from .alchemical_functions import alchemify as alchemify
from .base_strategy import AlchemicalStateCreationStrategy

logger = logging.getLogger(__name__)


class SireCreationStrategy(AlchemicalStateCreationStrategy):
    """Strategy for creating alchemical states using Sire."""

    _TMP_DIR = "tmp_fes_ml_sire"

    def create_alchemical_state(
        self,
        top_file: str,
        crd_file: str,
        alchemical_atoms: List[int],
        lambda_lj: Union[float, None],
        lambda_q: Union[float, None],
        lambda_interpolate: Union[float, None],
        lambda_ml_correction: Union[float, None],
        lambda_emle: Union[float, None],
        minimise_iterations: int = 1,
        ml_potential: str = "ani2x",
        ml_potential_kwargs: Optional[Dict[str, Any]] = None,
        create_system_kwargs: Optional[Dict[str, Any]] = None,
        topology: Optional[_mm.app.topology.Topology] = None,
        dynamics_kwargs: Optional[Dict[str, Any]] = None,
        emle_kwargs: Optional[Dict[str, Any]] = None,
        integrator: Optional[Any] = None,
    ) -> AlchemicalState:
        """
        Create an alchemical state for the given lambda values using OpenMM Systems created with Sire.

        Parameters
        ----------
        top_file : str
            Path to the topology file.
        crd_file : str
            Path to the coordinate file.
        alchemical_atoms : list of int
            A list of atom indices to be alchemically modified.
        lambda_lj : float or None
            The lambda value for the softcore Lennard-Jones potential.
        lambda_q : float or None
            The lambda value to scale the charges.
        lambda_interpolate : float or None
            The lambda value to interpolate between the ML and MM potentials in a mechanical embedding scheme.
        lambda_emle : float or None
            The lambda value to interpolate between the ML and MM potentials in a electrostatic embedding scheme.
        minimise_iterations : int, optional, default=1
            The number of minimisation iterations to perform before creating the alchemical state.
            1 step is enough to bring the geometry to the distances imposed by the restraints.
            If None, no minimisation is performed.
        ml_potential : str, optional, default='ani2x'
            The machine learning potential to use in the mechanical embedding scheme.
        ml_potential_kwargs : dict, optional, default=None
            Additional keyword arguments to be passed to MLPotential when creating the ML potential in OpenMM-ML.
            See: https://openmm.github.io/openmm-ml/dev/generated/openmmml.MLPotential.html
        create_system_kwargs : dict, optional, default=None
            Additional keyword arguments to be passed to the createSystem or createMixedSystem methods of the
            OpenMM-ML package. See: https://openmm.github.io/openmm-ml/dev/generated/openmmml.models.macepotential.MACEPotentialImpl.html
        topology : openmm.app.Topology, optional, default=None
            The OpenMM topology object.
        dynamics_kwargs : dict
            Additional keyword arguments to be passed to sire.mol.Dynamics.
            See https://sire.openbiosim.org/api/mol.html#sire.mol.Dynamics.
        emle_kwargs : dict
            Additional keyword arguments to be passed to the EMLECalculator.
            See TODO.
        integrator : Any, optional, default=None
            The OpenMM integrator to use. If None, the integrator is the one used in the dynamics_kwargs, if provided.
            Otherwise, the default is a LangevinMiddle integrator with a 1 fs timestep and a 298.15 K temperature.

        Returns
        -------
        AlchemicalState
            The alchemical state.
        """
        if lambda_interpolate is not None and lambda_emle is not None:
            raise ValueError(
                "The lambda_interpolate and lambda_emle parameters are mutually exclusive."
            )

        if any([lambda_interpolate, lambda_emle]) and lambda_ml_correction is not None:
            raise ValueError(
                "The lambda_ml_correction parameter is not compatible with lambda_interpolate or lambda_emle."
            )

        if dynamics_kwargs is None:
            dynamics_kwargs = {
                "timestep": "1fs",
                "constraint": "none",
                "cutoff_type": "pme",
                "cutoff": "9A",
                "integrator": "langevin_middle",
                "temperature": "298.15K",
            }
        else:
            dynamics_kwargs = _deepcopy(dynamics_kwargs)

        if emle_kwargs is None:
            emle_kwargs = {"method": "electrostatic", "backend": "torchani"}
        else:
            emle_kwargs = _deepcopy(emle_kwargs)

        logger.debug("-" * 100)
        logger.debug("Creating alchemical state using SireCreationStrategy.")
        logger.debug(f"top_file: {top_file}")
        logger.debug(f"crd_file: {crd_file}")
        logger.debug(f"alchemical_atoms: {alchemical_atoms}")
        logger.debug(f"lambda_lj: {lambda_lj}")
        logger.debug(f"lambda_q: {lambda_q}")
        logger.debug(f"lambda_interpolate: {lambda_interpolate}")
        logger.debug(f"lambda_ml_correction: {lambda_ml_correction}")
        logger.debug(f"lambda_emle: {lambda_emle}")
        logger.debug(f"ml_potential: {ml_potential}")
        logger.debug(f"topology: {topology}")
        logger.debug("dynamics_kwargs:")
        for key, value in dynamics_kwargs.items():
            logger.debug(f"{key}: {value}")
        logger.debug("emle_kwargs:")
        for key, value in emle_kwargs.items():
            logger.debug(f"{key}: {value}")

        # Load the molecular system.
        mols = _sr.load(top_file, crd_file, show_warnings=True)

        # Select the QM subsystem
        alchemical_subsystem = mols.atoms(alchemical_atoms)

        # Write QM subsystem parm7 to a temporary file
        alchemical_prm7 = _sr.save(
            alchemical_subsystem,
            directory=self._TMP_DIR,
            filename="alchemical_subsystem.prm7",
            format=["prm7"],
        )

        # Write the full system parm7 to a temporary file
        parm7 = _sr.save(
            mols,
            directory=self._TMP_DIR,
            filename="full_system.prm7",
            format=["prm7"],
        )

        if lambda_emle is not None:
            if len(alchemical_subsystem) == len(mols.atoms()):
                raise ValueError(
                    "The QM subsystem cannot contain all atoms in the system. "
                    "Please select a subset of atoms to be treated with the QM method "
                    "or use 'lambda_interpolate' instead of 'lambda_emle'."
                )

            # MM Charges of the QM subsystem
            mm_charges = _np.asarray(
                [atom.charge().value() for atom in alchemical_subsystem]
            )

            # Set up the emle-engine calculator
            calculator = _EMLECalculator(
                lambda_interpolate=lambda_emle,
                qm_indices=alchemical_atoms,
                mm_charges=mm_charges,
                parm7=parm7[0],
                **emle_kwargs,
            )

            # Create an EMLEEngine bound to the calculator using the same cutoff as the dynamics
            mols, engine = _sr.qm.emle(
                mols, alchemical_subsystem, calculator, dynamics_kwargs["cutoff"], 20
            )

            # Add the QM engine to the dynamics kwargs
            # Set the perturbable constraint to none to avoid having bond lenghts close to zero
            dynamics_kwargs["qm_engine"] = engine
            dynamics_kwargs["perturbable_constraint"] = "none"

            # If CustomNonbondedForce are present in the system (e.g. created by Sire when using EMLE),
            # the use_dispersion_correction should be set to False to avoid errors such as:
            # CustomNonbondedForce: Long range correction did not converge.  Does the energy go to 0 faster than 1/r^2?
            # Once these forces are removed, the use_dispersion_correction can be set to True for the NonbondedForce
            disp_correction = dynamics_kwargs.get("map", {}).get(
                "use_dispersion_correction", False
            )
            if disp_correction:
                dynamics_kwargs["map"]["use_dispersion_correction"] = False

        # Create a QM/MM dynamics object
        d = mols.dynamics(**dynamics_kwargs)

        if minimise_iterations:
            d.minimise(minimise_iterations)

        # Get the underlying OpenMM context.
        omm = d._d._omm_mols

        # Get the OpenMM system.
        system = omm._system

        if lambda_emle is not None:
            # Remove unnecessary forces from the system that are set by Sire
            forces = system.getForces()
            for i, force in reversed(list(enumerate(forces))):
                if isinstance(force, _mm.CustomBondForce) or isinstance(
                    force, _mm.CustomNonbondedForce
                ):
                    system.removeForce(i)

            # Once the CustomNonbondedForce are removed,
            # go back to the original use_dispersion_correction value

            if disp_correction:
                forces = system.getForces()
                for force in forces:
                    if isinstance(force, _mm.NonbondedForce):
                        force.setUseDispersionCorrection(True)
                        dynamics_kwargs["map"]["use_dispersion_correction"] = True

        # Remove contraints from the alchemical atoms
        # TODO: Make this optional
        for i in range(system.getNumConstraints() - 1, -1, -1):
            p1, p2, _ = system.getConstraintParameters(i)
            if p1 in alchemical_atoms or p2 in alchemical_atoms:
                system.removeConstraint(i)

        # Alchemify the system
        system = alchemify(
            system=system,
            alchemical_atoms=alchemical_atoms,
            lambda_lj=lambda_lj,
            lambda_q=lambda_q,
            lambda_interpolate=lambda_interpolate,
            lambda_ml_correction=lambda_ml_correction,
            ml_potential=ml_potential,
            ml_potential_kwargs=ml_potential_kwargs,
            create_system_kwargs=create_system_kwargs,
            topology=_app.AmberPrmtopFile(alchemical_prm7[0]).topology,
        )

        if integrator is None:
            # Create a new integrator
            integrator = _deepcopy(omm.getIntegrator())
        else:
            integrator = _deepcopy(integrator)

        # Create a new context and set positions and velocities
        topology = _app.AmberPrmtopFile(parm7[0]).topology
        simulation = _mm.app.Simulation(topology, system, integrator, omm.getPlatform())
        simulation.context.setPositions(omm.getState(getPositions=True).getPositions())

        try:
            simulation.context.setVelocitiesToTemperature(integrator.getTemperature())
        except AttributeError:
            simulation.context.setVelocitiesToTemperature(
                float(dynamics_kwargs["temperature"][:-1])
            )

        logger.debug("Energy decomposition of the system:")
        logger.debug(
            f"Total potential energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}"
        )
        energy_decomp = energy_decomposition(system, simulation.context)
        for force, energy in energy_decomp.items():
            logger.debug(f"{force}: {energy}")
        logger.debug("Alchemical state created successfully.")
        logger.debug("-" * 100)

        # Create the AlchemicalState
        alc_state = AlchemicalState(
            system=system,
            context=simulation.context,
            integrator=integrator,
            simulation=simulation,
            topology=topology,
            lambda_lj=lambda_lj,
            lambda_q=lambda_q,
            lambda_interpolate=lambda_interpolate,
            lambda_emle=lambda_emle,
            lambda_ml_correction=lambda_ml_correction,
        )

        # Clean up the temporary directory
        _shutil.rmtree(self._TMP_DIR)

        return alc_state
