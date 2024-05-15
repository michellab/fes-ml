"""Free Energy Simulation (FES) module."""
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import openmm as _mm
import openmm.unit as _unit

from .alchemical import AlchemicalState
from .alchemical import alchemical_factory as alchemical_factory

logger = logging.getLogger(__name__)


class FES:
    """
    A class to perform free energy calculations.

    Attributes
    ----------
    lambda_schedule : dict
        Dictionary with the λ values for the alchemical states.
    alchemical_states : list of AlchemicalState
        List of alchemical states.
    alchemical_atoms : list of int
        List of atom indices to be alchemically modified.
    output_prefix : str
        Prefix for the output files.
    checkpoint_frequency : int
        Frequency to save the state of the object.
    checkpoint_file : str
        Path to the checkpoint file.
    write_frame_frequency : int
        Frequency (in number of iterations) to write the frames to a DCD file.
    _iter : int
        Current iteration.
    _alc_id : int
        Current alchemical state id.
    _positions : postions vectors
        Current positions.
    _pbc : pbc vectors
        Current periodic box vectors.
    _U_kl : list of list float
        Potential energies sampled at a single state k and evaluated at all states l.
    _U_kln : list of list of list float
        Potential energies sampled at a all states k and evaluated at all states l.
    _create_alchemical_states_args : tuple
        Arguments to recreate the alchemical states upon deserialization.
    _create_alchemical_states_kwargs : dict
        Keyword arguments to recreate the alchemical states upon deserialization.
    _force_groups : dict
        Dictionary with the force group for each force.
    """

    __serializables__ = [
        "lambda_schedule",
        "alchemical_atoms",
        "output_prefix",
        "checkpoint_frequency",
        "checkpoint_file",
        "write_frame_frequency",
        # Private variables
        "_iter",
        "_alc_id",
        "_positions",
        "_pbc",
        "_U_kl",
        "_U_kln",
        "_create_alchemical_states_args",
        "_create_alchemical_states_kwargs",
        "_force_groups",
    ]

    def __init__(
        self,
        lambda_schedule: Optional[dict] = None,
        alchemical_atoms: Optional[List[int]] = None,
        output_prefix: str = "fes",
        checkpoint_frequency: int = 100,
        checkpoint_file: Optional[str] = None,
        write_frame_frequency: int = 0,
        restart: bool = False,
    ) -> None:
        """
        Initialize the FES object.

        Parameters
        ----------
        lambda_schedule : dict, optional, default=None
            Dictionary with the λ values for the alchemical states.
            The keys of the dictionary are the names of the modifications
            and the values are lists of λ values.
        alchemical_atoms : list of int, optional, default=None
            List of atom indices to be alchemically modified.
        output_prefix : str, optional, default="fes"
            Prefix for the output files.
        checkpoint_frequency : int, optional, default=100
            Frequency to save the state of the object.
        checkpoint_file : str, optional, default=f"{output_prefix}.pickle"
            Path to the checkpoint file.
        write_frame_frequency : int, optional, default=0
            Frequency (in number of iterations) to write the frames to a DCD file.
            If 0, the frames are not written.
        restart : bool, optional, default=False
            Whether to restart from the last checkpoint.
        """
        self.lambda_schedule = lambda_schedule
        self.alchemical_atoms = alchemical_atoms
        self.alchemical_states: List[AlchemicalState] = None
        self.output_prefix = output_prefix
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_file = checkpoint_file or f"{output_prefix}.pickle"
        self.write_frame_frequency = write_frame_frequency

        # Checkpoint variables
        self._iter: Optional[int] = None
        self._alc_id: Optional[int] = None
        self._pos: Optional[Any] = None
        self._pbc: Optional[Any] = None
        self._U_kl: Optional[List[List[float]]] = None
        self._U_kln: Optional[List[List[List[float]]]] = None
        self._create_alchemical_states_args: Optional[Tuple[Any, ...]] = None
        self._create_alchemical_states_kwargs: Optional[Dict[str, Any]] = None
        self._force_groups: Optional[Dict[Any, int]] = None

        if restart:
            assert os.path.exists(
                self.checkpoint_file
            ), f"Checkpoint file {self.checkpoint_file} does not exist."
            logger.info(f"Simulation will restart from {self.checkpoint_file}")
            with open(self.checkpoint_file, "rb") as f:
                state = pickle.load(f)
            self.__setstate__(state)

    def __getstate__(self) -> Dict[str, Any]:
        """Get the state of the object."""
        state = {key: getattr(self, key) for key in self.__serializables__}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the object."""
        for key, value in state.items():
            setattr(self, key, value)

        # Recreate the alchemical states
        self.alchemical_states = []
        self.create_alchemical_states(
            self.alchemical_atoms,
            self.lambda_schedule,
            *self._create_alchemical_states_args,
            **self._create_alchemical_states_kwargs,
        )

        for alc in self.alchemical_states:
            assert isinstance(
                alc, AlchemicalState
            ), "alchemical_state must be an AlchemicalState object."
            alc.check_integrity()
            alc.context.setPositions(self._positions)
            alc.context.setPeriodicBoxVectors(*self._pbc)

        # Set the force groups
        if self._force_groups is not None:
            self.set_force_groups(force_group_dict=self._force_groups)

    def _save_state(self) -> None:
        """Save the state of the object."""
        state = self.__getstate__()
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved state to {self.checkpoint_file}")

    def create_alchemical_states(
        self,
        lambda_schedule: Dict[str, List[Union[float, None]]],
        alchemical_atoms: List[int] = None,
        modifications_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        strategy_name: str = "sire",
        *args,
        **kwargs,
    ) -> List[AlchemicalState]:
        """
        Create a batch of alchemical states for the given λ values.

        Parameters
        ----------
        lambda_schedule : dict
            Dictionary with the λ values for the alchemical states.
        alchemical_atoms : list
            List of atom indices to be alchemically modified.
        modifications_kwargs : dict
            A dictionary of keyword arguments for the modifications.
            It is structured as follows:
            {
                "modification_name": {
                    "key1": value1,
                    "key2": value2,
                    ...
                },
                ...
            }
        strategy_name : str, optional, default='sire'
            The name of the strategy to create the alchemical state.
            Available strategies are: "sire".

        Returns
        -------
        alchemical_states : list of AlchemicalState
            List of alchemical states.
        """
        # Store the arguments and keyword arguments to recreate the alchemical states
        # when the object is deserialized
        self._create_alchemical_states_args = args
        self._create_alchemical_states_kwargs = kwargs
        self.lambda_schedule = lambda_schedule
        self.alchemical_atoms = alchemical_atoms

        # Check that that each parameter has the same number of λ values
        nstates_param = [
            len(lambda_schedule.get(lambda_param, []))
            for lambda_param in lambda_schedule
        ]

        nstates = nstates_param[0]
        if not all([n == nstates for n in nstates_param]):
            raise ValueError("All λ parameters must have the same number of λ values.")

        self.alchemical_states = []
        for i in range(nstates):
            alchemical_state = alchemical_factory.create_alchemical_state(
                strategy_name,
                alchemical_atoms=alchemical_atoms,
                lambda_schedule={
                    k: v[i] for k, v in lambda_schedule.items() if v[i] is not None
                },
                modifications_kwargs=modifications_kwargs,
                *args,
                **kwargs,
            )

            self.alchemical_states.append(alchemical_state)

        return self.alchemical_states

    def minimize_batch(
        self,
        tolerance: _unit.Quantity = 10 * _unit.kilojoules_per_mole / _unit.nanometer,
        max_iterations: int = 0,
        reporter: Any = None,
    ) -> List[AlchemicalState]:
        """
        Minimize a batch of alchemical states.

        Parameters
        ----------
        tolerance : openmm.unit.Quantity
            The energy tolerance to which the system should be minimized.
        max_iterations : int, optional, default=0
            The maximum number of iterations to run the minimization.
        reporter : OpenMM Minimization Report, optional, default=None
            A reporter object to report the minimization progress.
            See http://docs.openmm.org/latest/api-python/generated/openmm.openmm.MinimizationReporter.html

        Returns
        -------
        alchemical_states : list of AlchemicalState
            List of alchemical states.
        """
        assert (
            self.alchemical_states is not None
        ), "The alchemical states have not been created. Run `create_alchemical_states` first."

        for alc in self.alchemical_states:
            self.minimize_state(tolerance, max_iterations, alc, reporter=reporter)

        return self.alchemical_states

    def minimize_state(
        self,
        tolerance: _unit.Quantity = 10 * _unit.kilojoules_per_mole / _unit.nanometer,
        max_iterations: int = 0,
        alchemical_state: AlchemicalState = None,
        window: int = None,
        reporter: Any = None,
    ) -> AlchemicalState:
        """
        Run a single minimization.

        Parameters
        ----------
        tolerance : openmm.unit.Quantity
            The energy tolerance to which the system should be minimized.
        max_iterations : int, optional, default=0
            The maximum number of iterations to run the minimization.
        alchemical_state : AlchemicalState
            Alchemical state to minimize.
        reporter: OpenMM Minimization Report, optional, default=None
            A reporter object to report the minimization progress.
            See http://docs.openmm.org/latest/api-python/generated/openmm.openmm.MinimizationReporter.html

        Returns
        -------
        alchemical_state : AlchemicalState
            Alchemical state.
        """
        # Initial checks
        if alchemical_state is None:
            if window is None:
                raise ValueError("Either alchemical_state or window must be provided.")
            else:
                alchemical_state = self.alchemical_states[window]

        assert isinstance(
            alchemical_state, AlchemicalState
        ), "alchemical_state must be an AlchemicalState object."
        alchemical_state.check_integrity()

        logger.info(
            f"Minimizing {alchemical_state} with tolerance {tolerance} and max_iterations {max_iterations}"
        )
        _mm.LocalEnergyMinimizer.minimize(
            alchemical_state.context, tolerance, max_iterations, reporter
        )

        return alchemical_state

    def equilibrate_batch(self, nsteps: int) -> List[AlchemicalState]:
        """
        Equilibrate batch of alchemical states..

        Parameters
        ----------
        nsteps : int
            Number of steps to run each equilibration.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            List of alchemical states.
        """
        assert (
            self.alchemical_states is not None
        ), "The alchemical states have not been created. Run `create_alchemical_states` first."

        for alc in self.alchemical_states:
            self.equilibrate_state(nsteps, alc)

        return self.alchemical_states

    def equilibrate_state(
        self, nsteps: int, alchemical_state: AlchemicalState = None, window: int = None
    ) -> AlchemicalState:
        """
        Equilibrate a single alchemical state.

        Parameters
        ----------
        nsteps : int
            Number of steps to run the equilibration.
        alchemical_state : AlchemicalState
            Alchemical state to equilibrate.
        window : int
            Window index. alchemical_state takes precedence over window.

        Returns
        -------
        alchemical_state : AlchemicalState
            Alchemical state.
        """
        if alchemical_state is None:
            if window is None:
                raise ValueError("Either alchemical_state or window must be provided.")
            else:
                alchemical_state = self.alchemical_states[window]

        assert isinstance(
            alchemical_state, AlchemicalState
        ), "alchemical_state must be an AlchemicalState object."
        alchemical_state.check_integrity()

        logger.info(f"Equilibrating {alchemical_state}")
        alchemical_state.simulation.step(nsteps)

        return alchemical_state

    def run_single_state(
        self,
        niterations: int,
        nsteps: int,
        alchemical_state: AlchemicalState = None,
        window: int = None,
        reporters: Optional[List[Any]] = None,
    ) -> List[List[float]]:
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
        reporters : list of OpenMM reporters, optional, default=None
            List of reporters to append to the simulation.

        Returns
        -------
        U_kl : list of list float
            Potential energies of the sampled configurations evaluated for the various alchemical states.
        """
        if alchemical_state is None:
            if window is None:
                raise ValueError("Either alchemical_state or window must be provided.")
            else:
                alchemical_state = self.alchemical_states[window]

        # Check the integrity of the alchemical state
        assert isinstance(
            alchemical_state, AlchemicalState
        ), "alchemical_state must be an AlchemicalState object."
        alchemical_state.check_integrity()

        if not self._U_kl:
            self._iter = 0
            self._U_kl = [[] for _ in range(len(self.alchemical_states))]

        logger.info(
            f"Running production for {alchemical_state} with {niterations} iterations and {nsteps} steps per iteration"
        )

        try:
            integrator_temperature = alchemical_state.integrator.getTemperature()
        except AttributeError:
            integrator_temperature = (
                alchemical_state.integrator.temperature * _unit.kelvin
            )

        kT = (
            _unit.AVOGADRO_CONSTANT_NA
            * _unit.BOLTZMANN_CONSTANT_kB
            * integrator_temperature
        )

        # Append reporters to the simulation
        if reporters is not None:
            for reporter in reporters:
                alchemical_state.simulation.reporters.append(reporter)

        for iteration in range(self._iter, niterations):
            logger.info(f"{alchemical_state} iteration {iteration + 1} / {niterations}")

            alchemical_state.simulation.step(nsteps)
            self._positions = alchemical_state.context.getState(
                getPositions=True
            ).getPositions()
            self._pbc = alchemical_state.context.getState().getPeriodicBoxVectors()

            # Compute energies at all alchemical states
            for alc_id, alc_ in enumerate(self.alchemical_states):
                alc_.context.setPositions(self._positions)
                alc_.context.setPeriodicBoxVectors(*self._pbc)
                self._U_kl[alc_id].append(
                    alc_.context.getState(getEnergy=True).getPotentialEnergy() / kT
                )

            # Save the checkpoint
            if iteration % self.checkpoint_frequency == 0 and iteration > 0:
                self._iter = iteration + 1
                self._save_state()

        return self._U_kl

    def run_production_batch(
        self, niterations: int, nsteps: int, reporters: Optional[List[Any]] = None
    ) -> List[List[List[float]]]:
        """
        Run simulations for all alchemical states.

        Parameters
        ----------
        niterations : int
            Number of iterations to run the simulations.
        nsteps : int
            Number of steps per iteration.
        reporters : list of OpenMM reporters, optional, default=None
            List of reporters to append to the simulation.

        Returns
        -------
        u_kln : np.ndarray
            Array with the potential energies of the simulations.
        """
        assert (
            self.alchemical_states is not None
        ), "The alchemical states have not been created. Run `create_alchemical_states` first."

        # Resume from the last checkpoint
        if not self._U_kln:
            self._alc_id = 0
            self._U_kln = []

        logger.info(
            f"Starting production run with {self.alchemical_states[self._alc_id]} with id {self._alc_id}"
        )
        for alc_id in range(self._alc_id, len(self.alchemical_states)):
            alc = self.alchemical_states[alc_id]
            self._U_kln.append(
                self.run_single_state(niterations, nsteps, alc, reporters=reporters)
            )
            self._alc_id = alc_id + 1
            self._iter = 0
            self._save_state()

        return self._U_kln
