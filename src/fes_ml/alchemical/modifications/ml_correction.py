"""Module for the MLCorrectionModification class and its factory."""

import logging
from typing import List

import openmm as _mm

from .base_modification import BaseModification, BaseModificationFactory
from .ml_base_modification import MLBaseModification
from .ml_potential import MLPotentialModification

logger = logging.getLogger(__name__)


class MLCorrectionModificationFactory(BaseModificationFactory):
    """Factory for creating MLCorrectionModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of MLCorrectionModification.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        MLCorrectionModification
            The modification to be applied.
        """
        return MLCorrectionModification(*args, **kwargs)


class MLCorrectionModification(MLBaseModification, BaseModification):
    """Class to add a CustomCVForce that is a Δ ML correction."""

    NAME = "MLCorrection"
    pre_dependencies: List[str] = [MLPotentialModification.NAME]

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
        This code is heavily inspired on this https://github.com/openmm/openmm-ml/blob/main/openmmml/mlpotential.py#L190-L351.
        """
        # Create the CustomCVForce
        cv, mm_vars, ml_vars, _, ml_forces = self.create_cv(
            system, alchemical_atoms, lambda_value, *args, **kwargs
        )

        # Remove all ML forces from the system as we are interested in the correction
        forces_to_remove = sorted([force_id for force_id, _ in ml_forces], reverse=True)
        for force_id in forces_to_remove:
            system.removeForce(force_id)

        # Set the energy function
        ml_sum = "+".join(ml_vars) if len(ml_vars) > 0 else "0"
        mm_sum = "+".join(mm_vars) if len(mm_vars) > 0 else "0"
        ml_interpolation_function = f"lambda_interpolate*({ml_sum} - ({mm_sum}))"
        cv.setEnergyFunction(ml_interpolation_function)
        cv.setName(self.modification_name)
        system.addForce(cv)

        logger.debug(f"ML correction function: {ml_interpolation_function}")

        return system
