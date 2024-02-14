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

    def run_simulation_batch(self, niterations, nsteps):
        """
        Run a batch of simulations.

        Parameters
        ----------
        niterations : int
            Number of iterations to run the simulations.

        nsteps : int
            Number of steps to run the simulations.

        Returns
        -------
        u_kln : np.ndarray
            Array with the potential energies of the simulations.
        """
        if len(self.alchemical_states) == 0:
            raise ValueError("No alchemical states were found.")

        nstates = len(self.alchemical_states)
        u_kln = np.zeros([nstates, nstates, niterations], np.float64)

        kT = (
            unit.AVOGADRO_CONSTANT_NA
            * unit.BOLTZMANN_CONSTANT_kB
            * nstates[0].integrator.getTemperature()
        )

        U_kn = [[] for k in range(nstates)]
        for k, alc in enumerate(self.alchemical_states):
            if not isinstance(alc, AlchemicalState):
                raise ValueError(
                    f"Expected alchemical state, but got {type(alc)} instead."
                )

            for iteration in range(niterations):
                print(f"{alc} iteration {iteration} / {niterations}")
                alc.integrator.step(nsteps)
                positions = alc.context.getState(getPositions=True).getPositions()
                pbc = alc.context.getState().getPeriodicBoxVectors()

                # Compute energies at all alchemical states
                for l, alc_ in enumerate(self.alchemical_states):
                    alc_.context.setPositions(positions)
                    alc_.context.setPeriodicBoxVectors(*pbc)
                    u_kln[k, l, iteration] = (
                        alc_.context.getState(getEnergy=True).getPotentialEnergy() / kT
                    )

        np.save("u_kln.npy", u_kln)

        return U_kn
