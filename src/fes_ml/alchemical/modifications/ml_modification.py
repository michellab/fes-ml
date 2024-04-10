"""This module contains the implementation of the LJSoftCoreModification class."""

import logging
from typing import Any, Dict, List, Optional

import openmm as _mm
import openmm.app as _app

from ..alchemist import Alchemist
from .base_modification import BaseModification, BaseModificationFactory

logger = logging.getLogger(__name__)


class MLModificationFactory(BaseModificationFactory):
    """Factory for creating MLModifaction instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of MLModification.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        MLModification
            The modification to be applied.
        """
        return MLModification(*args, **kwargs)


class MLModification(BaseModification):
    NAME = "lambda_interpolate"

    def apply(
        self,
        system: _mm.System,
        lambda_value: float,
        alchemical_atoms: List[int],
        ml_potential: Any,
        topology: _app.topology.Topology,
        interpolate: bool = True,
        create_system_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Create an alchemical System that is partly modeled with a ML potential and partly
        with a conventional force field.

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
        interpolate : bool, optional, default=True
            If True, the System will include Forces to compute the energy both with the
            conventional force field and with this potential, and to smoothly interpolate
            between them.  If False, the System will include only the Forces to compute
            the energy with this potential.
        create_system_kwargs : dict, optional, default=None
            Additional keyword arguments to pass to the createMixedSystem method of the ML potential.

        Returns
        -------
        system : openmm.System
            The modified System.

        Notes
        -----
        The CustomCVForce defines a global parameter called "lambda_interpolate" that interpolates
        between the two potentials.  When lambda_interpolate=0, the energy is computed entirely with
        the conventional force field.  When lambda_interpolate=1, the energy is computed entirely with
        the ML potential.  You can set its value by calling setParameter() on the Context.
        """
        if create_system_kwargs is None:
            create_system_kwargs = {}

        system = ml_potential.createMixedSystem(
            topology,
            system,
            alchemical_atoms,
            interpolate=interpolate,
            **create_system_kwargs,
        )

        # Get the CustomCVForce that interpolates between the two potentials and set its global parameter
        # TODO: generalise this to work in cases where there are multiple CustomCVForces
        forces = {force.__class__.__name__: force for force in system.getForces()}
        cv_force = forces["CustomCVForce"]
        cv_force.setGlobalParameterDefaultValue(0, lambda_value)

        return system


# Register the LJSoftCoreModificationFactory with the Alchemist
Alchemist.register_modification_factory(MLModification.NAME, MLModificationFactory())
