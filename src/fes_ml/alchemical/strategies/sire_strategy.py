"""Sire alchemical state creation strategy."""

import json
from copy import deepcopy as _deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as _np
import openmm as _mm
import sire as _sr
from emle.calculator import EMLECalculator as _EMLECalculator

from ...log import logger
from ...utils import energy_decomposition as energy_decomposition
from ..alchemical_state import AlchemicalState
from .alchemical_functions import alchemify as alchemify
from .base_strategy import AlchemicalStateCreationStrategy


class SireCreationStrategy(AlchemicalStateCreationStrategy):
    """Strategy for creating alchemical states using Sire."""

    def create_alchemical_state(
        self,
        top_file: str,
        crd_file: str,
        alchemical_atoms: List[int],
        lambda_lj: Union[float, None],
        lambda_q: Union[float, None],
        lambda_interpolate: Union[float, None],
        lambda_emle: Union[float, None],
        ml_potential: str = "ani2x",
        topology: _mm.app.Topology = None,
        dynamics_kwargs: Optional[Dict[str, Any]] = None,
        emle_kwargs: Optional[Dict[str, Any]] = None,
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
        ml_potential : str, optional, default='ani2x'
            The machine learning potential to use in the mechanical embedding scheme.
        topology : openmm.app.Topology, optional, default=None
            The OpenMM topology object.
        dynamics_kwargs : dict
            Additional keyword arguments to be passed to sire.mol.Dynamics.
            See https://sire.openbiosim.org/api/mol.html#sire.mol.Dynamics.
        emle_kwargs : dict
            Additional keyword arguments to be passed to the EMLECalculator.
            See TODO.

        Returns
        -------
        AlchemicalState
            The alchemical state.
        """
        if lambda_interpolate is not None and lambda_emle is not None:
            raise ValueError(
                "Only one of lambda_interpolate and lambda_emle can be not None at the same time."
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

        logger.debug("*" * 40)
        logger.debug("Creating alchemical state using SireCreationStrategy.")
        logger.debug(f"top_file: {top_file}")
        logger.debug(f"crd_file: {crd_file}")
        logger.debug(f"alchemical_atoms: {alchemical_atoms}")
        logger.debug(f"lambda_lj: {lambda_lj}")
        logger.debug(f"lambda_q: {lambda_q}")
        logger.debug(f"lambda_interpolate: {lambda_interpolate}")
        logger.debug(f"lambda_emle: {lambda_emle}")
        logger.debug(f"ml_potential: {ml_potential}")
        logger.debug(f"topology: {topology}")
        logger.debug(f"dynamics_kwargs: {json.dumps(dynamics_kwargs, indent=4)}")
        logger.debug(f"emle_kwargs: {json.dumps(emle_kwargs, indent=4)}")

        # Load the molecular system.
        mols = _sr.load(top_file, crd_file, show_warnings=True)

        if lambda_emle is not None:
            # Select the QM subsystem
            qm_subsystem = mols.atoms(alchemical_atoms)

            if len(qm_subsystem) == len(mols.atoms()):
                raise ValueError(
                    "The QM subsystem cannot contain all atoms in the system. "
                    "Please select a subset of atoms to be treated with the QM method "
                    "or use 'lambda_interpolate' instead of 'lambda_emle'."
                )

            # Write QM subsystem parm7 to a temporary file
            parm7 = _sr.save(
                qm_subsystem,
                directory="tmp",
                filename="qm_subsystem.prm7",
                format=["prm7"],
            )

            # MM Charges of the QM subsystem
            mm_charges = _np.asarray([atom.charge().value() for atom in qm_subsystem])

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
                mols, qm_subsystem, calculator, dynamics_kwargs["cutoff"], 20
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
            # we go back to the original use_dispersion_correction value
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
            ml_potential=ml_potential,
            topology=topology,
        )

        # Create a new integrator
        integrator = omm.getIntegrator().__copy__()

        # Create a new context and set positions and velocities
        context = _mm.Context(system, integrator, omm.getPlatform())
        context.setPositions(omm.getState(getPositions=True).getPositions())
        context.setVelocitiesToTemperature(integrator.getTemperature())

        logger.debug(
            f"Potential energy: {context.getState(getEnergy=True).getPotentialEnergy()}"
        )
        energy_decomp = energy_decomposition(system, context)
        for force, energy in energy_decomp.items():
            logger.debug(f"{force}: {energy}")
        logger.debug("Alchemical state created successfully.")
        logger.debug("*" * 40)

        # Create the AlchemicalState
        alc_state = AlchemicalState(
            system=system,
            context=context,
            integrator=integrator,
            lambda_lj=lambda_lj,
            lambda_q=lambda_q,
            lambda_interpolate=lambda_interpolate,
            lambda_emle=lambda_emle,
        )

        return alc_state
