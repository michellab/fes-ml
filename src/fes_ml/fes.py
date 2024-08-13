"""Free Energy Simulation (FES) module."""
import logging
import os
import pickle
from copy import deepcopy as _deepcopy
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
    alchemical_states : list of AlchemicalState
        List of alchemical states.
    output_prefix : str
        Prefix for the output files.
    checkpoint_frequency : int
        Frequency to save the state of the object.
    checkpoint_file : str
        Path to the checkpoint file.
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
    _lambda_schedule : dict
        Dictionary with the λ values for the alchemical states.
    _alchemical_atoms : list of int
        List of atom indices to be alchemically modified.
    """

    __non_serializables__ = ["alchemical_states"]

    def __init__(
        self,
        output_prefix: str = "fes",
        checkpoint_frequency: int = 100,
        checkpoint_file: Optional[str] = None,
        restart: bool = True,
        recreate_alchemical_states: bool = False,
    ) -> None:
        """
        Initialize the FES object.

        Parameters
        ----------
        output_prefix : str, optional, default="fes"
            Prefix for the output files.
        checkpoint_frequency : int, optional, default=100
            Frequency to save the state of the FES object.
        checkpoint_file : str, optional, default=f"{output_prefix}.pickle"
            Path to the checkpoint file.
        write_frame_frequency : int, optional, default=0
            Frequency (in number of iterations) to write the frames to a DCD file.
            If 0, the frames are not written.
        restart : bool, optional, default=False
            Whether to restart from the last checkpoint.
        recreate_alchemical_states : bool, optional, default=False
            Whether to recreate the alchemical states upon deserialization.
        """
        self.alchemical_states: List[AlchemicalState] = None
        self.output_prefix = output_prefix
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_file = checkpoint_file or f"{output_prefix}.pickle"

        # Checkpoint variables
        self._iter: Optional[int] = None
        self._alc_id: Optional[int] = None
        self._pos: Optional[Any] = None
        self._pbc: Optional[Any] = None
        self._U_kl: Optional[List[List[float]]] = None
        self._U_kln: Optional[List[List[List[float]]]] = None

        # Variables to recreate the alchemical states
        self._lambda_schedule: Dict[str, List[Union[float, None]]] = None
        self._alchemical_atoms: List[int] = None
        self._create_alchemical_states_args: Optional[Tuple[Any, ...]] = None
        self._create_alchemical_states_kwargs: Optional[Dict[str, Any]] = None

        # Automatically load the state if restart is True
        if os.path.exists(self.checkpoint_file) and restart:
            self._load_state(recreate_alchemical_states=recreate_alchemical_states)
        elif restart and not os.path.exists(self.checkpoint_file):
            logger.warning(
                f"Attempted to restart from {self.checkpoint_file} but the file does not exist. "
                "Restarting from scratch."
            )

    def __getstate__(self) -> Dict[str, Any]:
        """Get the state of the object."""
        state = {
            key: getattr(self, key)
            for key in self.__dict__
            if key not in self.__non_serializables__
        }
        return state

    def __setstate__(
        self, state: Dict[str, Any], recreate_alchemical_states: bool = False
    ) -> None:
        """Set the state of the object."""
        for key, value in state.items():
            setattr(self, key, value)

        if recreate_alchemical_states:
            self.alchemical_states = []
            self.create_alchemical_states(
                self._lambda_schedule,
                self._alchemical_atoms,
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

    def _load_state(self, recreate_alchemical_states: bool) -> None:
        """Reload the state of the object."""
        logger.info(f"Simulation will restart from {self.checkpoint_file}")
        with open(self.checkpoint_file, "rb") as f:
            state = pickle.load(f)
        self.__setstate__(state, recreate_alchemical_states)

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
        self._lambda_schedule = lambda_schedule
        self._alchemical_atoms = alchemical_atoms

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

    def set_velocities(
        self,
        temperature: Union[float, _unit.Quantity],
        window: Optional[int] = None,
        alchemical_state: Optional[AlchemicalState] = None,
    ) -> List[AlchemicalState]:
        """
        Set the velocities of the alchemical states.

        Parameters
        ----------
        temperature : openmm.unit.Quantity or float
            Temperature of the system in Kelvin.
        window : int, optional, default=None
            Window index.
        alchemical_state : AlchemicalState
            Alchemical state to set the velocities.
        """
        if isinstance(temperature, _unit.Quantity):
            temperature = temperature.value_in_unit(_unit.kelvin)
        elif not isinstance(temperature, float):
            raise ValueError("Temperature must be a float or a Quantity.")

        if not window and not alchemical_state:
            for alc in self.alchemical_states:
                alc.context.setVelocitiesToTemperature(temperature)
        else:
            if alchemical_state is None:
                alchemical_state = self.alchemical_states[window]

            alchemical_state.context.setVelocitiesToTemperature(temperature)

        return self.alchemical_states

    def _minimize_state(
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

    def minimize(
        self,
        tolerance: _unit.Quantity = 10 * _unit.kilojoules_per_mole / _unit.nanometer,
        max_iterations: int = 0,
        reporter: Any = None,
        window: int = None,
        alchemical_state: AlchemicalState = None,
    ) -> List[AlchemicalState]:
        """
        Minimize

        Parameters
        ----------
        tolerance : openmm.unit.Quantity
            The energy tolerance to which the system should be minimized.
        max_iterations : int, optional, default=0
            The maximum number of iterations to run the minimization.
        reporter : OpenMM Minimization Report, optional, default=None
            A reporter object to report the minimization progress.
            See http://docs.openmm.org/latest/api-python/generated/openmm.openmm.MinimizationReporter.html
        window : int, optional, default=None
            Window index.
        alchemical_state : AlchemicalState
            Alchemical state to minimize.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            List of alchemical states.
        """
        if window is None and alchemical_state is None:
            # Attempt batch minimization
            assert (
                self.alchemical_states is not None
            ), "The alchemical states have not been created. Run `create_alchemical_states` first."

            for alc in self.alchemical_states:
                self._minimize_state(tolerance, max_iterations, alc, reporter=reporter)
        else:
            self._minimize_state(
                tolerance, max_iterations, alchemical_state, window, reporter
            )

        return self.alchemical_states

    def _equilibrate_state(
        self,
        nsteps: int,
        alchemical_state: AlchemicalState = None,
        window: int = None,
        reporters: Optional[List[Any]] = None,
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
        reporters : list of OpenMM reporters, optional, default=None
            List of reporters to append to the simulation.

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

        # Append reporters to the simulation
        if reporters is not None:
            for reporter in reporters:
                alchemical_state.simulation.reporters.append(reporter)

        logger.info(f"Equilibrating {alchemical_state}")
        alchemical_state.simulation.step(nsteps)

        return alchemical_state

    def equilibrate(
        self,
        nsteps: int,
        alchemical_state: AlchemicalState = None,
        window: int = None,
        reporters: Optional[List[Any]] = None,
    ) -> List[AlchemicalState]:
        """
        Equilibrate batch of alchemical states..

        Parameters
        ----------
        nsteps : int
            Number of steps to run each equilibration.
        window : int, optional, default=None
            Window index.
        alchemical_state : AlchemicalState
            Alchemical state to equilibrate.
        reporters : list of OpenMM reporters, optional, default=None
            List of reporters to append to the simulation.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            List of alchemical states.
        """
        if window is None and alchemical_state is None:
            # Attempt batch equilibration
            assert (
                self.alchemical_states is not None
            ), "The alchemical states have not been created. Run `create_alchemical_states` first."

            for alc in self.alchemical_states:
                self._equilibrate_state(nsteps, alc, reporters=reporters)
        else:
            self._equilibrate_state(
                nsteps, alchemical_state, window, reporters=reporters
            )

        return self.alchemical_states

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
        else:
            # We are resuming from a checkpoint, so we need to set the initial state
            alchemical_state.simulation.loadState(f"{self.output_prefix}_openmm.chk")

        logger.info(
            f"Running production for {alchemical_state} with {niterations} iterations and {nsteps} steps per iteration"
        )

        try:
            integrator_temperature = alchemical_state.integrator.getTemperature()
        except AttributeError:
            integrator_temperature = alchemical_state.integrator.temperature

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
            if iteration % (self.checkpoint_frequency - 1) == 0 and iteration > 0:
                self._iter = iteration + 1
                self._save_state()
                alchemical_state.simulation.saveState(
                    f"{self.output_prefix}_openmm.chk"
                )

        tmp_U_kl = _deepcopy(self._U_kl)
        self._U_kl = None

        return tmp_U_kl

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
