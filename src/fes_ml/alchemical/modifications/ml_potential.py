"""Module for the MLPotentialModification class and its factory."""

import logging
from typing import List, Optional

import openmm as _mm
import openmm.app as _app

from .base_modification import BaseModification, BaseModificationFactory
from .intramolecular import (IntraMolecularBondedRemovalModification,
                             IntraMolecularNonBondedExceptionsModification,
                             IntraMolecularNonBondedForcesModification)
from .ml_base_modification import MLBaseModification

logger = logging.getLogger(__name__)


class MLPotentialModificationFactory(BaseModificationFactory):
    """Factory for creating MLModifaction instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """
        Create an instance of MLPotentialModification.

        Returns
        -------
        MLPotentialModification
            The modification to be applied.
        """
        return MLPotentialModification(*args, **kwargs)


class MLPotentialModification(MLBaseModification, BaseModification):
    """Class to add a ML potential to the System."""

    NAME = "MLPotential"
    pre_dependencies: List[str] = [IntraMolecularNonBondedForcesModification.NAME]
    post_dependencies: List[str] = [
        IntraMolecularBondedRemovalModification.NAME,
        IntraMolecularNonBondedExceptionsModification.NAME,
    ]

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        topology: _app.Topology,
        name: str = "ani2x",
        modelPath: Optional[str] = None,
        forceGroup: int = 0,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Add a ML potential to the System.

        Parameters
        ----------
        system : openmm.System
            The System to modify.
        alchemical_atoms : list of int
            The indices of the atoms to model with the ML potential.
        topology : openmm.app.Topology
            The Topology of the System.
        name : str
            The name of the ML potential to use.
        modelPath : str
            The path to the deployed model.
        name : str
            The name of the ML potential to use.
        modelPath : str
            The path to the deployed model.
        forceGroup : int
            The force group to assign the ML potential forces to.

        Returns
        -------
        system : openmm.System
            The modified System.
        """
        from openmmml import MLPotential

        mlp = MLPotential(name=name, modelPath=modelPath)
        mlp._impl.addForces(topology, system, alchemical_atoms, forceGroup, *args, **kwargs)
        return system
