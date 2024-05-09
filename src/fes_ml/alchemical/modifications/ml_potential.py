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


class MLPotentialModification(BaseModification):
    NAME = "MLPotential"
    pre_dependencies: List[str] = [IntraMolecularNonBondedForcesModification.NAME]
    post_dependencies: List[str] = [
        IntraMolecularBondedRemovalModification.NAME,
        IntraMolecularNonBondedExceptionsModification.NAME,
    ]

    _ANI_POTENTIALS = ["ani2x"]
    _MACE_POTENTIALS = [
        "mace",
        "mace-off23-small",
        "mace-off23-medium",
        "mace-off23-large",
    ]

    _ANI_POTENTIALS = ["ani2x"]
    _MACE_POTENTIALS = [
        "mace",
        "mace-off23-small",
        "mace-off23-medium",
        "mace-off23-large",
    ]

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        topology: _app.Topology,
        name: str = "ani2x",
        modelPath: Optional[str] = None,
        name: str = "ani2x",
        modelPath: Optional[str] = None,
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

        Returns
        -------
        system : openmm.System
            The modified System.
        """

        if name in self._ANI_POTENTIALS:
            self._create_ani2x(topology, system, alchemical_atoms, *args, **kwargs)
        elif name in self._MACE_POTENTIALS:
            self._create_mace(
                name, topology, system, alchemical_atoms, modelPath, *args, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown ML potential: {name}. Currently supported potentials are {self._ANI_POTENTIALS + self._MACE_POTENTIALS}"
                f"Unknown ML potential: {name}. Currently supported potentials are {self._ANI_POTENTIALS + self._MACE_POTENTIALS}"
            )

        return system

    def _create_ani2x(
        self, topology, system, alchemical_atoms, forceGroup: int = 0, *args, **kwargs
    ) -> _mm.System:
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
        forceGroup : int
            The force group to assign the ANI-2x potential.
        args : tuple
            Additional arguments to pass to the addForces method of the ANI-2x potential.
        kwargs : dict
            Additional keyword arguments to pass to the addForces method of the ANI-2x potential.

        Returns
        -------
        system : openmm.System
            The modified System.
        """
        ani = anipotential.ANIPotentialImpl(name="ani2x")
        ani.addForces(topology, system, alchemical_atoms, forceGroup, *args, **kwargs)

        return system

    def _create_mace(
        self,
        name: str,
        topology,
        system,
        alchemical_atoms,
        modelPath: Optional[str] = None,
        forceGroup: int = 0,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Add a MACE potential to the System.

        Parameters
        ----------
        name : str
            The name of the MACE potential to use.
        name : str
            The name of the MACE potential to use.
        topology : openmm.app.Topology
            The Topology of the System.
        system : openmm.System
            The System to modify.
        alchemical_atoms : list of int
            The indices of the atoms to model with the MACE potential.
        modelPath : str
            The path to the MACE model.
        forceGroup : int
            The force group to assign the MACE potential.
        args : tuple
            Additional arguments to pass to the addForces method of the MACE potential.
        kwargs : dict
            Additional keyword arguments to pass to the addForces method of the MACE potential.

        Returns
        -------
        system : openmm.System
            The modified System.
        """
        macepotiml = macepotential.MACEPotentialImpl(name=name, modelPath=modelPath)
        macepotiml.addForces(
            topology, system, alchemical_atoms, forceGroup, *args, **kwargs
        )

        return system
