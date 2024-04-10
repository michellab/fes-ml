"""This module contains the implementation of the LJSoftCoreModification class."""

import logging
import re
from typing import Any, Dict, List, Optional

import openmm as _mm
import openmm.app as _app

from ..alchemist import Alchemist
from .base_modification import BaseModification, BaseModificationFactory

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


class MLCorrectionModification(BaseModification):
    NAME = "lambda_ml_corr"

    def apply(
        self,
        system: _mm.System,
        lambda_value: float,
        alchemical_atoms: List[int],
        ml_potential: Any,
        topology: _app.topology.Topology,
        create_system_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Create an ML correction to the energy of the System.

        Parameters
        ----------
        system : openmm.System
            The System to modify.
        lambda_value : float
            The value of the alchemical state parameter.
        alchemical_atoms : list of int
            The indices of the atoms to model with the ML potential.
        ml_potential : openmmml.mlpotential.MLPotential
            The ML potential to use.
        topology : openmm.app.topology.Topology
            The Topology of the System.
        create_system_kwargs : dict, optional, default=None
            Additional keyword arguments to pass to the createMixedSystem method of the ML potential.

        Returns
        -------
        system : openmm.System
            The modified System.

        Notes
        -----
        The CustomCVForce defines a global parameter called "lambda_ml_corr" that controls the
        amount of ML correction to apply.
        """

        if create_system_kwargs is None:
            create_system_kwargs = {}

        # Create a copy of the original system
        system = system.__deepcopy__()

        system_tmp = ml_potential.createMixedSystem(
            topology,
            system,
            alchemical_atoms,
            interpolate=True,
            **create_system_kwargs,
        )

        # Get the CustomCVForce that interpolates between the two potentials and set its global parameter
        forces = {force.__class__.__name__: force for force in system_tmp.getForces()}
        cv_force = forces["CustomCVForce"]
        cv_force.setName("MLCorrForce")
        cv_force.addGlobalParameter(f"{MLCorrectionModification.NAME}", lambda_value)

        # Get the ML and MM terms from the energy expression
        energy_expression = cv_force.getEnergyFunction()
        ml_forces = re.findall(r"\b(mlForce\w*)\b", energy_expression)
        mm_forces = re.findall(r"\b(mmForce\w*)\b", energy_expression)

        # Define the correction energy expression
        corr_energy_expression = (
            f"{MLCorrectionModification.NAME}*(("
            + "+".join(ml_forces)
            + ") - ("
            + "+".join(mm_forces)
            + "))"
        )
        cv_force.setEnergyFunction(corr_energy_expression)
        cv_force.setGlobalParameterDefaultValue(0, lambda_value)

        # Copy back the modified force to the original system
        system.addForce(cv_force)

        return system


# Register the LJSoftCoreModificationFactory with the Alchemist
Alchemist.register_modification_factory(
    MLCorrectionModification.NAME, MLCorrectionModificationFactory()
)
