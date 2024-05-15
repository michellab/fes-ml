"""Module that contains the MTS class with various utility functions for the Multiple Timestep (MTS) integrator."""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import openmm as _mm
import openmm.unit as _unit

from .alchemical import AlchemicalState

logger = logging.getLogger(__name__)


class MTS:
    """Class with various utility functions for the Multiple Timestep (MTS) integrator."""

    def __init__(
        self, alchemical_states: Optional[List[AlchemicalState]] = None
    ) -> None:
        """Initialize the MTS object."""
        self.alchemical_states = alchemical_states
        self._force_groups: Dict[str, int] = None

    def create_integrator(
        self,
        dt: Union[float, _unit.Quantity],
        groups: List[Tuple[int, int]],
        temperature: Union[float, _unit.Quantity] = 298.15 * _unit.kelvin,
        friction: Union[float, _unit.Quantity] = 1.0 / _unit.picosecond,
        type: str = "langevin",
    ) -> Any:
        """
        Create the MTS integrator.

        For more information on the MTS integrator, see the OpenMM documentation:
        - Langevin: http://docs.openmm.org/latest/api-python/generated/openmm.mtsintegrator.MTSLangevinIntegrator.html
        - MTS: http://docs.openmm.org/7.0.0/api-python/generated/simtk.openmm.mtsintegrator.MTSIntegrator.html

        Parameters
        ----------
        dt
            The largest (outermost) integration timestep.
        groups
            A list of tuples defining the force groups.
            The first element of each tuple is the force group index,
            and the second element is the number of times that force
            group should be evaluated in one time step.
        temperature
            The temperature of the system.
        friction
            The friction coefficient.
        type
            The type of integrator. Options are "langevin" or "mts".


        Returns
        -------
        integrator : Any
            MTS integrator.
        """
        if type.lower() == "langevin":
            if isinstance(temperature, float):
                temperature = temperature * _unit.kelvin
            elif not isinstance(temperature, _unit.Quantity):
                raise ValueError("Temperature must be a float or a Quantity.")

            if isinstance(friction, float):
                friction = friction / _unit.picosecond
            elif not isinstance(friction,  _unit.Quantity):
                raise ValueError("Friction must be a float or a Quantity.")
            
            integrator = _mm.MTSLangevinIntegrator(
                temperature,
                friction,
                dt,
                groups,
            )
        elif type.lower() == "mts":
            integrator = _mm.MTSIntegrator(dt, groups)
        else:
            raise ValueError(f"Invalid MTS integrator type: {type}")

        return integrator

    def set_force_groups(
        self,
        alchemical_states: Optional[List[AlchemicalState]] = None,
        slow_forces: Optional[List[Any]] = None,
        fast_force_group: int = 0,
        slow_force_group: int = 1,
        force_group_dict: Optional[Dict[Any, int]] = None,
    ) -> Dict[Any, int]:
        """
        Set the force groups for the alchemical states.

        Parameters
        ----------
        alchemical_states : list of AlchemicalState, optional, default=None
            List of alchemical states to set the force groups.
            If not provided, the alchemical states set in the object initialization are used.
        slow_forces : list of names of OpenMM forces, optional, default=None
            List of forces to be considered slow.
        fast_force_group : int
            Force group for the fast forces.
        slow_force_group : int, optional, default=1
            Force group for the slow forces.
        force_group_dict : dict, optional, default=None
            Dictionary with the force group for each force.
            If provided, it takes precedence over the other arguments and the force groups are set
            according to the dictionary. Otherwise, the force groups are set according to the other arguments.

        Returns
        -------
        force_groups : dict
            Dictionary with the force group for each force.

        Notes
        -----
        The force groups are set as follows:
        - If `force_group_dict` is provided, the force groups are set according to the dictionary.
        All forces not in the dictionary are set to `fast_force_group`.
        - If `slow_forces` is provided, the forces in `slow_forces` are set to `slow_force_group` and
        the others to `fast_force_group`.
        - If `slow_forces` is not provided, all forces are set to `fast_force_group`.
        """
        if alchemical_states is not None:
            assert isinstance(
                alchemical_states, list
            ), "alchemical_states must be a list."
            for alc in alchemical_states:
                assert isinstance(
                    alc, AlchemicalState
                ), "All elements in alchemical_states must be AlchemicalState."
                alc.check_integrity()
            self.alchemical_states = alchemical_states
        else:
            assert (
                self.alchemical_states is not None
            ), "alchemical_states must be provided to set the force groups."

        if force_group_dict is not None:
            for state in self.alchemical_states:
                unassigned_slow_forces = list(force_group_dict.keys())

                for force in state.system.getForces():
                    try:
                        force.setForceGroup(force_group_dict[force.getName()])
                        unassigned_slow_forces.remove(force.getName())
                    except KeyError:
                        force.setForceGroup(fast_force_group)

                if unassigned_slow_forces:
                    logger.warning(
                        f"Warning: the following forces {unassigned_slow_forces} were not found in {state}."
                    )

                state.context.reinitialize(preserveState=True)
        else:
            if slow_forces is not None:
                for state in self.alchemical_states:
                    unassigned_slow_forces = list(slow_forces)

                    for force in state.system.getForces():
                        if force.getName() in slow_forces:
                            force.setForceGroup(slow_force_group)
                            unassigned_slow_forces.remove(force.getName())
                        else:
                            force.setForceGroup(fast_force_group)

                    if unassigned_slow_forces:
                        logger.warning(
                            f"Warning: the following forces {unassigned_slow_forces} were not found in {state}."
                        )

                    state.context.reinitialize(preserveState=True)
            else:
                for state in self.alchemical_states:
                    for force in state.system.getForces():
                        force.setForceGroup(fast_force_group)

                    state.context.reinitialize(preserveState=True)

        # Store the force groups
        self._force_groups = {
            force.getName(): force.getForceGroup()
            for force in self.alchemical_states[0].system.getForces()
        }

        logger.info("Force groups set to:")
        for force, group in self._force_groups.items():
            logger.info(f"{force}: {group}")

        return self._force_groups
