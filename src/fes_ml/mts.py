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
        self._groups: List[Tuple[int, int]] = None

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
            0 should be the slowest force group.
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
            elif not isinstance(friction, _unit.Quantity):
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

        self._groups = groups

        return integrator

    def set_force_groups(
        self,
        alchemical_states: Optional[List[AlchemicalState]] = None,
        force_group_dict: Optional[Dict[Any, int]] = None,
        set_reciprocal_space_force_groups: Optional[int] = None,
    ) -> Dict[Any, int]:
        """
        Set the force groups for the alchemical states.

        Parameters
        ----------
        alchemical_states : list of AlchemicalState, optional, default=None
            List of alchemical states to set the force groups.
            If not provided, the alchemical states set in the object initialization are used.
        force_group_dict : dict, optional, default=None
            Dictionary with the force group for each force.
        set_reciprocal_space_force_groups: int, Optional, default=None
            Which group to set the reciprocal space force group to. By default, this is not set seperately.

        Notes
        -----
        - If a group is not provided, all forces are set to the fastest force group.
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
            # get the fastest force group for the unassigned forces
            fastest_group = max(self._groups, key=lambda fg_dt: fg_dt[1])[0]

            for state in self.alchemical_states:
                unassigned_dict_forces = list(force_group_dict.keys())

                for force in state.system.getForces():
                    try:
                        force.setForceGroup(force_group_dict[force.getName()])
                        unassigned_dict_forces.remove(force.getName())
                    except KeyError:
                        force.setForceGroup(fastest_group)

                if unassigned_dict_forces:
                    logger.warning(
                        f"Warning: the following forces {unassigned_dict_forces} were not found in {state}."
                    )

                # set the reciprocal force group to the defined group
                if set_reciprocal_space_force_groups is not None:
                    if isinstance(force, _mm.NonbondedForce):
                        force.setReciprocalSpaceForceGroup(
                            set_reciprocal_space_force_groups
                        )

                state.context.reinitialize(preserveState=True)

            force_groups = {
                            force.getName(): force.getForceGroup()
                            for force in state.system.getForces()
                            }
            logger.info(f"Force groups for state index {self.alchemical_states.index(state)} set to:")
            for force, group in force_groups.items():
                logger.info(f"{force}: {group}")
