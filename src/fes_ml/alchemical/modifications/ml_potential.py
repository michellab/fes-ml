"""This module contains the implementation of the LJSoftCoreModification class."""

import logging
from typing import Any, Dict, List, Optional

import openmm as _mm
import openmm.app as _app
from openmmml.models import anipotential, macepotential

from .base_modification import BaseModification, BaseModificationFactory
from .intramolecular import (
    IntraMolecularBondedRemovalModification,
    IntraMolecularNonBondedExceptionsModification,
    IntraMolecularNonBondedForcesModification,
)

logger = logging.getLogger(__name__)


class MLModificationFactory(BaseModificationFactory):
    """Factory for creating MLModifaction instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """
        Create an instance of MLModification.

        Returns
        -------
        MLModification
            The modification to be applied.
        """
        return MLModification(*args, **kwargs)


class MLModification(BaseModification):
    NAME = "MLPotential"    
    pre_dependencies: List[str] = [IntraMolecularNonBondedForcesModification.NAME]
    post_dependencies: List[str] = [IntraMolecularBondedRemovalModification.NAME,
                                    IntraMolecularNonBondedExceptionsModification.NAME]

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        ml_potential: str,
        topology: _app.Topology,
        addForces_kwargs: Optional[Dict[str, Any]] = None,
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
        ml_potential : openmmml.mlpotential.MLPotential
            The ML potential to use.
        topology : openmm.app.Topology
            The Topology of the System.

        Returns
        -------
        system : openmm.System
            The modified System.
        """
        addForces_kwargs = addForces_kwargs or {}

        if ml_potential == "ani2x":
            self._create_ani2x(topology, system, alchemical_atoms, addForces_kwargs)
        elif ml_potential == "mace":
            self._create_mace(topology, system, alchemical_atoms, addForces_kwargs)
        else:
            raise ValueError(f"Unknown ML potential: {ml_potential}. Currently supported potentials are 'ani2x' and 'mace'.")


        return system
    
    def _create_ani2x(self, topology, system, alchemical_atoms, addForces_kwargs) -> _mm.System:
        """
        Add an ANI-2x potential to the System.

        Parameters
        ----------
        topology : openmm.app.Topology
            The Topology of the System.
        system : openmm.System
            The System to modify.
        alchemical_atoms : list of int
            The indices of the atoms to model with the ANI-2x potential.
        addForces_kwargs : dict
            Additional keyword arguments to pass to the addForces method of the ANI-2x potential.
        
        Returns
        -------
        system : openmm.System
            The modified System.
        """
        ani = anipotential.ANIPotentialImpl(name="ani2x")
        ani.addForces(topology, system, alchemical_atoms, 0, **addForces_kwargs)

        return system

    def _create_mace(self, topology, system, alchemical_atoms, addForces_kwargs) -> _mm.System:
        """
        Add a MACE potential to the System.

        Parameters
        ----------
        topology : openmm.app.Topology
            The Topology of the System.
        system : openmm.System
            The System to modify.
        alchemical_atoms : list of int
            The indices of the atoms to model with the MACE potential.
        addForces_kwargs : dict
            Additional keyword arguments to pass to the addForces method of the MACE potential.
        
        Returns
        -------
        system : openmm.System
            The modified System.
        """
        macepotiml = macepotential.MACEPotentialImpl(name="mace-off23-small", modelPath="")
        macepotiml.addForces(topology, system, alchemical_atoms, **addForces_kwargs)

        return system
