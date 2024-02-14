"""Sire alchemical state creation strategy."""
from typing import Any, Dict, List, Optional, Union

import numpy as _np
import openmm as _mm
import sire as _sr
from emle.calculator import EMLECalculator as _EMLECalculator

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
                "cutoff": "12A",
                "integrator": "langevin_middle",
                "temperature": "298.15K",
            }

        if emle_kwargs is None:
            emle_kwargs = {"method": "electrostatic", "backend": "torchani"}

        # Load the molecular system.
        mols = _sr.load(top_file, crd_file, show_warnings=True)

        if lambda_emle is not None:
            # Select the QM subsystem
            qm_subsystem = mols.atoms(alchemical_atoms)

            # Write QM subsystem parm7 to a temporary file
            parm7 = _sr.save(
                qm_subsystem,
                directory="cache",
                filename="qm_subsystem.pdb",
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

            # Create an EMLEEngine bound to the calculator
            mols, engine = _sr.qm.emle(mols, qm_subsystem, calculator)

            # Create a ML/MM dynamics object
            d = mols.dynamics(qm_engine=engine, **dynamics_kwargs)
        else:
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

        # Alchemify the system
        system = alchemify(
            system=system,
            alchemical_atoms=alchemical_atoms,
            lambda_lj=lambda_lj,
            lambda_q=lambda_q,
            lambda_interpolate=lambda_interpolate,
            ml_potential=None,
            topology=None,
        )

        # Create a new integrator
        integrator = omm.getIntegrator().__copy__()

        # Create a new context and set positions and velocities
        context = _mm.Context(system, integrator, omm.getPlatform())
        context.setPositions(omm.getState(getPositions=True).getPositions())
        context.setVelocitiesToTemperature(integrator.getTemperature())

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
