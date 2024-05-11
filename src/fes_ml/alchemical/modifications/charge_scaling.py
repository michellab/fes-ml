"""Module for the ChargeScalingModification class and its factory."""

import logging
from typing import List

import openmm as _mm

from .base_modification import BaseModification, BaseModificationFactory
from .intramolecular import (
    IntraMolecularNonBondedExceptionsModification,
    IntraMolecularNonBondedForcesModification,
)

logger = logging.getLogger(__name__)


class ChargeScalingModificationFactory(BaseModificationFactory):
    """Factory for creating ChargeScalingModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """
        Create an instance of ChargeScalingModification.

        Returns
        -------
        ChargeScalingModification
            The modification to be applied.
        """
        return ChargeScalingModification(*args, **kwargs)


class ChargeScalingModification(BaseModification):
    """Class to scale the charges of the alchemical atoms in the System."""

    NAME = "ChargeScaling"
    pre_dependencies: List[str] = [IntraMolecularNonBondedForcesModification.NAME]
    post_dependencies: List[str] = [IntraMolecularNonBondedExceptionsModification.NAME]

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        lambda_value: float,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Scale the charges of the alchemical atoms in the System.

        Only the intermolecular interactions are affected by this modification.

        Parameters
        ----------
        system : openmm.System
            The System to modify.
        lambda_value : float
            The value of the alchemical state parameter.
        alchemical_atoms : list of int
            The indices of the alchemical atoms in the System.

        Returns
        -------
        system : openmm.System
            The modified System with the charges scaled.
        """
        nb_forces = [
            force
            for force in system.getForces()
            if isinstance(force, _mm.NonbondedForce)
        ]
        if len(nb_forces) > 1:
            raise ValueError(
                "The system must not contain more than one NonbondedForce."
            )
        elif len(nb_forces) == 0:
            logger.warning(
                "The system does not contain a NonbondedForce and therefore no charge scaling will be applied.=."
            )
            return system
        else:
            force = nb_forces[0]
            for index in range(system.getNumParticles()):
                [charge, sigma, epsilon] = force.getParticleParameters(index)
                if index in alchemical_atoms:
                    # Scale the charges of the alchemical atoms
                    force.setParticleParameters(
                        index, charge * lambda_value, sigma, epsilon
                    )

        return system
