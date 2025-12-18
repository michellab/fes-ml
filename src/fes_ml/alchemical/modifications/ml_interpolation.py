"""Module for the MLInterpolationModification class and its factory."""

import logging
from typing import List

import openmm as _mm

from .base_modification import BaseModification, BaseModificationFactory
from .ml_base_modification import MLBaseModification
from .ml_potential import MLPotentialModification
from .intramolecular import (
    IntraMolecularBondedRemovalModification,
    IntraMolecularNonBondedExceptionsModification,
)

logger = logging.getLogger(__name__)


class MLInterpolationModificationFactory(BaseModificationFactory):
    """Factory for creating MLInterpolationModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of MLInterpolationModification.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        MLInterpolationModification
            The modification to be applied.
        """
        return MLInterpolationModification(*args, **kwargs)


class MLInterpolationModification(MLBaseModification, BaseModification):
    """Class to add a CustomCVForce to interpolate between ML and MM forces."""

    NAME = "MLInterpolation"
    pre_dependencies = [MLPotentialModification.NAME]
    post_dependencies: List[str] = [
        IntraMolecularBondedRemovalModification.NAME,
        IntraMolecularNonBondedExceptionsModification.NAME,
    ]

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        lambda_value: float,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Apply the ML interpolation modification to the system.

        Parameters
        ----------
        system : openmm.System
            The system to be modified.
        alchemical_atoms : list of int
            The indices of the alchemical atoms in the system.
        lambda_value : float
            The value of the alchemical state parameter.
        args : tuple
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        openmm.System
            The modified system.
        Notes
        -----
        This code is heavily inspired on https://github.com/openmm/openmm-ml/blob/main/openmmml/mlpotential.py#L190-L351.
        """
        # Create the CustomCVForce
        cv, mm_vars, ml_vars, _, ml_forces = self.create_cv(system, alchemical_atoms, lambda_value, *args, **kwargs)

        # Remove ML forces from the system
        forces_to_remove = sorted([force_id for force_id, _ in ml_forces], reverse=True)
        for force_id in forces_to_remove:
            system.removeForce(force_id)

        # Set the energy function
        ml_sum = "+".join(ml_vars) if len(ml_vars) > 0 else "0"
        mm_sum = "+".join(mm_vars) if len(mm_vars) > 0 else "0"
        ml_interpolation_function = f"lambda_interpolate*({ml_sum}) + (1-lambda_interpolate)*({mm_sum})"
        cv.setEnergyFunction(ml_interpolation_function)
        system.addForce(cv)
        cv.setName(self.modification_name)

        logger.debug(f"ML interpolation function: {ml_interpolation_function}")

        return system
