from dataclasses import dataclass, field

import openmm as mm


@dataclass
class AlchemicalState:
    """
    A dataclass representing the alchemical state of a molecular system.

    Attributes
    ----------
    lambda_lj : float
        The lambda value for the softcore Lennard-Jones potential.
    lambda_q : float
        The lambda value to scale the charges.
    lambda_interpolate : float
        The lambda value to interpolate between the ML and MM potentials in a mechanical embedding scheme.
        If lambda_interpolate=1, the alchemical subsystem is fully described by the ML potential.
        If lambda_interpolate=0, the alchemical subsystem is fully described by the MM potential.
    lambda_emle : float
        The lambda value to interpolate between the ML and MM potentials in a electrostatic embedding scheme.
        If lambda_emle=1, the alchemical subsystem is fully described by the ML potential.
        If lambda_emle=0, the alchemical subsystem is fully described by the MM potential.
    system : openmm.System
        The OpenMM system associated with the alchemical state.
    context : openmm.Context
        The OpenMM context associated with the alchemical state.
    """

    lambda_lj: float = field(repr=True, default=None)
    lambda_q: float = field(repr=True, default=None)
    lambda_interpolate: float = field(repr=True, default=None)
    lambda_emle: float = field(repr=True, default=None)
    system: mm.System = field(repr=False, default=None)
    context: mm.Context = field(repr=False, default=None)
