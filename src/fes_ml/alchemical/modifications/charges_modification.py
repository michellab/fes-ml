"""This module contains the implementation of the LJSoftCoreModification class."""

import logging
from typing import List

import openmm as _mm

from ..alchemist import Alchemist
from .base_modification import BaseModification, BaseModificationFactory

logger = logging.getLogger(__name__)


class ChargesModificationFactory(BaseModificationFactory):
    """Factory for creating ChargesModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of ChargesModification.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        ChargesModification
            The modification to be applied.
        """
        return ChargesModification(*args, **kwargs)


class ChargesModification(BaseModification):
    NAME = "lambda_q"

    def apply(
        self,
        system: _mm.System,
        lambda_value: float,
        alchemical_atoms: List[int],
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Scale the charges of the alchemical atoms in the System.

        Parameters
        ----------
        system : openmm.System
            The System to modify.
        lambda_value : float
            The value of the alchemical state parameter.
        alchemical_atoms : list of int
            The indices of the alchemical atoms in the System.
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        system : openmm.System
            The modified System with the charges scaled.
        """
        logging.info(f"Applying charges modification with lambda value: {lambda_value}")
        forces = {force.__class__.__name__: force for force in system.getForces()}
        nb_force = forces["NonbondedForce"]

        for index in range(system.getNumParticles()):
            [charge, sigma, epsilon] = nb_force.getParticleParameters(index)
            if index in alchemical_atoms:
                # Scale the charges of the alchemical atoms
                nb_force.setParticleParameters(
                    index, charge * lambda_value, sigma, epsilon
                )

        return system


Alchemist.register_modification_factory(
    ChargesModification.NAME, ChargesModificationFactory()
)
