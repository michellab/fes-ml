"""Module that defines the AlchemicalState dataclass."""

from dataclasses import dataclass, field
from typing import Dict, Union

import openmm as _mm
import openmm.app as _app


@dataclass
class AlchemicalState:
    """
    A dataclass representing the alchemical state of a molecular system.

    Attributes
    ----------
    system : openmm.System
        The OpenMM system associated with the alchemical state.
    context : openmm.Context
        The OpenMM context associated with the alchemical state.
    integrator : openmm.Integrator
        The OpenMM integrator associated with the alchemical state.
    simulation : openmm.app.Simulation
        The OpenMM simulation associated with the alchemical state.
    topology : openmm.app.Topology
        The OpenMM topology associated with the alchemical state.
    modifications : dict
        Dictionary mapping the name of the alchemical modifications
        applied to the system to the λ value.
    """

    # OpenMM objects
    system: _mm.System = field(repr=False, default=None)
    integrator: _mm.Integrator = field(repr=False, default=None)
    context: _mm.Context = field(repr=False, default=None)
    simulation: _app.Simulation = field(repr=False, default=None)
    topology: _mm.app.Topology = field(repr=False, default=None)

    # Alchemical state parameters
    modifications: Dict[str, Union[float, int]] = field(repr=True, default=None)

    def check_integrity(self):
        """
        Check the integrity of the alchemical state.

        Parameters
        ----------
        alchemical_state : AlchemicalState
            Alchemical state to check.
        """
        assert isinstance(self.system, _mm.System), "System is not an OpenMM System."
        assert isinstance(self.integrator, _mm.Integrator), "Integrator is not an OpenMM Integrator."
        assert isinstance(self.context, _mm.Context), "Context is not an OpenMM Context."
        assert isinstance(self.simulation, _app.Simulation), "Simulation is not an OpenMM Simulation."
        assert isinstance(self.topology, _mm.app.Topology), "Topology is not an OpenMM Topology."
        assert isinstance(self.modifications, dict), "Modifications is not a dictionary."
