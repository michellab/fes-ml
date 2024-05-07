import logging
from copy import deepcopy as _deepcopy

import openmm as _mm

from .base_modification import BaseModification, BaseModificationFactory

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


class MLInterpolationModification(BaseModification):
    NAME = "MLInterpolation"
    pre_dependencies = ["MLPotential"]
    post_dependencies = [
        "IntraMolecularNonBondedExceptions",
        "IntraMolecularBondedRemoval",
    ]

    def apply(
        self,
        system: _mm.System,
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

        cv = _mm.CustomCVForce("")
        cv.addGlobalParameter("MLInterpolation", lambda_value)

        # Add ML forces to the CV
        ml_forces = []
        for force_id, force in enumerate(system.getForces()):
            if force.getName() == "TorchForce":
                ml_forces.append((force_id, force))

        ml_vars = []
        for i, (force_id, force) in enumerate(ml_forces):
            name = f"mlForce{i+1}"
            cv.addCollectiveVariable(name, _deepcopy(force))
            ml_vars.append(name)

        # Add bonded forces to the CV
        bonded_forces = []
        for force in system.getForces():
            if (
                hasattr(force, "addBond")
                or hasattr(force, "addAngle")
                or hasattr(force, "addTorsion")
            ):
                bonded_forces.append(force)

        mm_vars = []
        for i, force in enumerate(bonded_forces):
            name = f"mmForce{i+1}"
            cv.addCollectiveVariable(name, _deepcopy(force))
            mm_vars.append(name)

        # Remove ML forces from the system
        # TODO: check if this is always necessary
        forces_to_remove = sorted([force_id for force_id, _ in ml_forces], reverse=True)
        for force_id in forces_to_remove:
            system.removeForce(force_id)

        # Set the energy function
        ml_sum = "+".join(ml_vars) if len(ml_vars) > 0 else "0"
        mm_sum = "+".join(mm_vars) if len(mm_vars) > 0 else "0"
        ml_interpolation_function = (
            f"MLInterpolation*({ml_sum}) + (1-MLInterpolation)*({mm_sum})"
        )
        cv.setEnergyFunction(ml_interpolation_function)
        system.addForce(cv)
        cv.setName(self.NAME)

        logger.debug(f"ML interpolation function: {ml_interpolation_function}")

        return system
