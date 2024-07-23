"""Module that contains the MLModificationBase from which ML modifications can inherit."""
from copy import deepcopy as _deepcopy
from typing import List, Tuple

import openmm as _mm

from .intramolecular import IntraMolecularBondedRemovalModification


class MLBaseModification:
    """
    MLModificationBase from which ML modifications can inherit.

    This class only contains utily functions and is not supposed to be used as a stand-alone modification.
    """

    @staticmethod
    def create_cv(
        system: _mm.System,
        alchemical_atoms: List[int],
        lambda_value: float,
        lambda_paramter_name: str = "lambda_interpolate",
        *args,
        **kwargs,
    ) -> Tuple[
        _mm.CustomCVForce, List[str], List[str], List[_mm.Force], List[_mm.Force]
    ]:
        """
        Create a CustomCVForce that will contain the ML and bonded MM forces.

        Parameters
        ----------
        system : openmm.System
            The system from which the forces will be extracted.
        alchemical_atoms : list of int
            The indices of the alchemical atoms.
        lambda_value : float
            The value of the lambda parameter.
        lambda_paramter_name : str
            The name of the lambda parameter.

        Returns
        -------
        CustomCVForce
            The CustomCVForce containing the ML and MM forces.
        list of str
            The names of the MM forces.
        list of str
            The names of the ML forces.
        list of openmm.Force
            The bonded MM forces.
        list of openmm.Force
            The ML forces.
        """
        cv = _mm.CustomCVForce("")
        cv.addGlobalParameter(lambda_paramter_name, lambda_value)

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
                # Remove bonded interactions between non-alchemical atoms
                force = _deepcopy(force)
                IntraMolecularBondedRemovalModification._remove_bonded_interactions(
                    force, alchemical_atoms, False
                )
                bonded_forces.append(force)

        mm_vars = []
        for i, force in enumerate(bonded_forces):
            name = f"{force.getName()}{i+1}"
            cv.addCollectiveVariable(name, _deepcopy(force))
            mm_vars.append(name)

        return cv, mm_vars, ml_vars, bonded_forces, ml_forces
