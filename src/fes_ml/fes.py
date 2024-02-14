from typing import Any, Dict, List

import numpy as np
import openmm.unit as unit

from .alchemical import AlchemicalState
from .alchemical import alchemical_factory as _alchemical_factory


class FES:
    """
    A class to perform free energy calculations.

    Attributes
    ----------
    crd_file : str
        Path to the coordinate file.
    top_file : str
        Path to the topology file.
    lambda_schedule : dict
        Dictionary with the lambda values for the alchemical states.
    alchemical_states : list of AlchemicalState
        List of alchemical states.
    """

    def __init__(
        self,
        crd_file: str,
        top_file: str,
        lambda_schedule: dict = None,
    ):
        """
        Initialize the FES object.

        Parameters
        ----------
        crd_file : str
            Path to the coordinate file.
        top_file : str
            Path to the topology file.
        lambda_schedule : dict, optional, default=None
            Dictionary with the lambda values for the alchemical states.
            The keys of the dictionary are the lambda parameters and the values are lists of lambda values.
            Available lambda parameters are: "lambda_lj", "lambda_q", "lambda_interpolate", "lambda_emle".
        """
        self.crd_file = crd_file
        self.top_file = top_file
        self.lambda_schedule = lambda_schedule
        self.alchemical_states: List[AlchemicalState] = []

    def create_alchemical_states(
        self,
        alchemical_atoms,
        lambda_schedule,
        *args,
        **kwargs,
    ) -> List[AlchemicalState]:
        """
        Create a batch of alchemical states for the given lambda values.

        Parameters
        ----------
        alchemical_atoms : list
            List of atom indices to be alchemically modified.
        lambda_schedule : dict
            Dictionary with the lambda values for the alchemical states.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            List of alchemical states.
        """
        # Check that that each parameter has the same number of lambda values
        nstates_param = [
            len(lambda_schedule.get(lambda_param, []))
            for lambda_param in lambda_schedule
        ]

        nstates = nstates_param[0]
        if not all([n == nstates for n in nstates_param]):
            raise ValueError(
                "All lambda parameters must have the same number of lambda values."
            )

        lambda_lj = lambda_schedule.get("lambda_lj", [None] * nstates)
        lambda_q = lambda_schedule.get("lambda_q", [None] * nstates)
        lambda_interpolate = lambda_schedule.get("lambda_interpolate", [None] * nstates)
        lambda_emle = lambda_schedule.get("lambda_emle", [None] * nstates)

        for i in range(nstates):
            alchemical_state = _alchemical_factory.create_alchemical_state(
                top_file=self.top_file,
                crd_file=self.crd_file,
                alchemical_atoms=alchemical_atoms,
                lambda_lj=lambda_lj[i],
                lambda_q=lambda_q[i],
                lambda_interpolate=lambda_interpolate[i],
                lambda_emle=lambda_emle[i],
                *args,
                **kwargs,
            )

            self.alchemical_states.append(alchemical_state)

        return self.alchemical_states

    def run_equilibration_batch(self, nsteps, minimize=True):
        """
        Run a batch of equilibrations.

        Parameters
        ----------
        nsteps : int
            Number of steps to run each equilibration.
        minimize : bool, optional, default=True
            If True, the energy will be minimized before running the equilibration.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            List of alchemical states.
        """
        if len(self.alchemical_states) == 0:
            raise ValueError("No alchemical states were found.")

        for alc in self.alchemical_states:
            print(f"Equilibrating {alc}")

            if minimize:
                alc.context.minimizeEnergy()

            alc.integrator.step(nsteps)

        print("Finished all equilibrations!")
        return self.alchemical_states

    def run_single_state(
        self,
        niterations: int,
        nsteps: int,
        alchemical_state: AlchemicalState = None,
        window: int = None,
    ):
        """
        Run a simulation for a given alchemical state or window index.

        Parameters
        ----------
        niterations : int
            Number of iterations to run the simulations.
        nsteps : int
            Number of steps per iteration.
        alchemical_state : AlchemicalState
            Alchemical state to run the simulation.
        window : int
            Window index. alchemical_state takes precedence over window.

        Returns
        -------
        U_kn : list of float
            Potential energies of the sampled configurations evaluated for the various alchemical states.
        """
        if alchemical_state is None:
            if window is None:
                raise ValueError("Either alchemical_state or window must be provided.")
            else:
                alchemical_state = self.alchemical_states[window]

        if not isinstance(alchemical_state, AlchemicalState):
            raise ValueError("alchemical_state must be an instance of AlchemicalState.")

        U_kn = []

        kT = (
            unit.AVOGADRO_CONSTANT_NA
            * unit.BOLTZMANN_CONSTANT_kB
            * alchemical_state.integrator.getTemperature()
        )

        for iteration in range(niterations):
            print(f"{alchemical_state} iteration {iteration} / {niterations}")
            alchemical_state.integrator.step(nsteps)
            positions = alchemical_state.context.getState(
                getPositions=True
            ).getPositions()
            pbc = alchemical_state.context.getState().getPeriodicBoxVectors()

            # Compute energies at all alchemical states
            for alc_ in self.alchemical_states:
                alc_.context.setPositions(positions)
                alc_.context.setPeriodicBoxVectors(*pbc)
                U_kn.append(
                    alc_.context.getState(getEnergy=True).getPotentialEnergy() / kT
                )

        return U_kn

    def run_production_batch(self, niterations, nsteps):
        """
        Run simulations for all alchemical states.

        Parameters
        ----------
        niterations : int
            Number of iterations to run the simulations.

        nsteps : int
            Number of steps per iteration.

        Returns
        -------
        u_kln : np.ndarray
            Array with the potential energies of the simulations.
        """
        if len(self.alchemical_states) == 0:
            raise ValueError("No alchemical states were found.")

        U_kln = []
        for alc in self.alchemical_states:
            U_kln.append(self.run_single_state(niterations, nsteps, alc))

        return U_kln
