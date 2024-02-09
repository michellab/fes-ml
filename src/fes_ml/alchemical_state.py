from dataclasses import dataclass, field
from typing import Dict

import openmm as mm
import openmm.app as app


@dataclass
class AlchemicalState:
    """
    A dataclass representing the alchemical state of a molecular system.

    Attributes
    ----------
    lambda_state : Dict[str, float]
        A dictionary representing the current values of alchemical parameters.
    system : openmm.System
        The OpenMM system.
    simulation : openmm.Simulation
        The OpenMM simulation.
    """

    lambda_state: Dict[str, float]
    system: mm.System = field(repr=False)
    simulation: app.Simulation = field(repr=False, default=None)
