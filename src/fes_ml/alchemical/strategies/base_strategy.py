"""Base strategy for alchemical state creation."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import openmm as _mm

from ...utils import energy_decomposition as energy_decomposition
from ..alchemical_state import AlchemicalState
from ..alchemist import Alchemist

logger = logging.getLogger(__name__)


class AlchemicalStateCreationStrategy(ABC):
    """Base class for alchemical state creation strategies."""

    @abstractmethod
    def create_alchemical_state(self, *args, **kwargs) -> AlchemicalState:
        """
        Create an alchemical state for the given λ values.

        Returns
        -------
        AlchemicalState
            The alchemical state.
        """
        pass

    def _run_alchemist(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        lambda_schedule: Dict[str, Union[float, int]],
        *args,
        **kwargs,
    ) -> None:
        """
        Run the alchemist to apply the alchemical modifications to the system.

        Parameters
        ----------
        lambda_schedule : dict
            A dicitonary containing a mapping of the alchemical modifications to the λ values.
        system : openmm.System
            The system to be modified.
        alchemical_atoms : list of int
            The list of alchemical atoms.
        args : list
            Additional arguments to be passed to the Alchemist ``apply_modifications`` method.
        kwargs : dict
            Additional keyword arguments to be passed to the Alchemist ``apply_modifications`` method.
        """
        alchemist = Alchemist()
        alchemist.create_alchemical_graph(lambda_schedule)
        alchemist.apply_modifications(
            system,
            alchemical_atoms,
            *args,
            **kwargs,
        )

    @staticmethod
    def _report_dict(
        dict_to_report: Dict[str, Any],
        dict_name: Optional[str] = None,
        indentation: int = 0,
        initial=True,
    ) -> None:
        if initial:
            logger.debug("")
            logger.debug("+" + "-" * 98 + "+")
        if dict_name is not None:
            logger.debug(f"{dict_name}")
            logger.debug("+" + "-" * 98 + "+")

        for key, value in dict_to_report.items():
            if isinstance(value, dict):
                logger.debug(f"{key}:")
                AlchemicalStateCreationStrategy._report_dict(
                    value, None, indentation + 4, False
                )
            else:
                logger.debug(f"{' '*indentation}{key}: {value}")

        if initial:
            logger.debug("+" + "-" * 98 + "+")

    @staticmethod
    def _report_energy_decomposition(context, system) -> None:
        """
        Report the energy decomposition of the system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM system.
        context : openmm.Context
            The OpenMM context.
        """
        logger.debug("")
        logger.debug("-" * 100)
        logger.debug("ENERGY DECOMPOSITION")
        logger.debug("-" * 100)
        logger.debug(
            f"Total potential energy: {context.getState(getEnergy=True).getPotentialEnergy()}"
        )
        energy_decomp = energy_decomposition(system, context)
        for force, energy in energy_decomp.items():
            logger.debug(f"{force}: {energy}")
        logger.debug("-" * 100)
        logger.debug("")

    @staticmethod
    def _remove_constraints(system: _mm.System, atoms: List[int]) -> _mm.System:
        """
        Remove constraints involving alchemical atoms from the system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM system.
        atoms : list of int
            The list of atoms to remove constraints from.

        Returns
        -------
        openmm.System
            The modified OpenMM system.
        """
        # Remove constraints involving alchemical atoms
        for i in range(system.getNumConstraints() - 1, -1, -1):
            p1, p2, _ = system.getConstraintParameters(i)
            if p1 in atoms or p2 in atoms:
                system.removeConstraint(i)

        return system
