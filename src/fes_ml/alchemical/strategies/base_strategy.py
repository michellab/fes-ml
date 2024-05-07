"""Base strategy for alchemical state creation."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

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
        Create an alchemical state for the given lambda values.

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
    ):
        """
        Run the alchemist to apply the alchemical modifications to the system.

        Parameters
        ----------
        lambda_schedule : dict
            A dicitonary containing a mapping of the alchemical modifications to the lambda values.
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
    def _report_creation_settings(passed_args: Dict[str, Any]) -> None:
        logger.debug("Creating alchemical state with the following parameters:")
        for key, value in passed_args.items():
            if isinstance(value, dict):
                logger.debug(f"{key}:")
                for k, v in value.items():
                    logger.debug(f"    {k}: {v}")
            else:
                logger.debug(f"{key}: {value}")

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
        logger.debug("-" * 50)
        logger.debug("ENERGY DECOMPOSITION")
        logger.debug("-" * 50)
        logger.debug(
            f"Total potential energy: {context.getState(getEnergy=True).getPotentialEnergy()}"
        )
        energy_decomp = energy_decomposition(system, context)
        for force, energy in energy_decomp.items():
            logger.debug(f"{force}: {energy}")
        logger.debug("-" * 50)
        logger.debug("")
