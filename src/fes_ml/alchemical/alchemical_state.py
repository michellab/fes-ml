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
