"""Free Energy Simulation (FES) module."""
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit

from .alchemical import AlchemicalState
from .alchemical import alchemical_factory as alchemical_factory

logger = logging.getLogger(__name__)


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
        "crd_file",
        "top_file",
        "sdf_file",
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
        crd_file: Optional[str] = None,
        top_file: Optional[str] = None,
        sdf_file: Optional[str] = None,
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
        crd_file : str
            Path to the coordinate file.
        top_file : str
            Path to the topology file.
        sdf_file : str
            Path to the sdf file. Used for the OpenMM creation strategy.
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
        self.crd_file = crd_file
        self.top_file = top_file
        self.sdf_file = sdf_file
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
            self._check_alchemical_state_integrity(alc)
            alc.context.setPositions(self._positions)
            alc.context.setPeriodicBoxVectors(*self._pbc)

        # Set the force groups
        if self._force_groups is not None:
            self.set_force_groups(force_group_dict=self._force_groups)

    @staticmethod
    def _check_alchemical_state_integrity(alchemical_state: AlchemicalState) -> None:
        """
        Check the integrity of the alchemical state.

        Parameters
        ----------
        alchemical_state : AlchemicalState
            Alchemical state to check.
        """
        assert isinstance(
            alchemical_state, AlchemicalState
        ), f"{alchemical_state} must be an instance of AlchemicalState."
        assert (
            alchemical_state.context is not None
        ), f"{alchemical_state} context is `None`."
        assert (
            alchemical_state.integrator is not None
        ), f"{alchemical_state} integrator is `None`."
        assert (
            alchemical_state.system is not None
        ), f"{alchemical_state} system is `None`."

    def _save_state(self) -> None:
        """Save the state of the object."""
        state = self.__getstate__()
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved state to {self.checkpoint_file}")

    def create_alchemical_states(
        self,
        lambda_schedule: Dict[str, List[Optional[float]]],
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
                top_file=self.top_file,
                crd_file=self.crd_file,
                sdf_file=self.sdf_file,
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

    def run_minimization_batch(
        self,
        tolerance: _unit.Quantity = 10 * _unit.kilojoules_per_mole / _unit.nanometer,
        max_iterations: int = 0,
        reporter: Any = None,
    ) -> List[AlchemicalState]:
        """
        Run a batch of minimizations.

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
            logger.info(
                f"Minimizing {alc} with tolerance {tolerance} and max_iterations {max_iterations}"
            )
            self._check_alchemical_state_integrity(alc)
            _mm.LocalEnergyMinimizer.minimize(
                alc.context, tolerance, max_iterations, reporter
            )

        return self.alchemical_states

    def run_equilibration_batch(self, nsteps: int) -> List[AlchemicalState]:
        """
        Run a batch of equilibrations.

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
            logger.info(f"Equilibrating {alc}")
            self._check_alchemical_state_integrity(alc)
            alc.simulation.step(nsteps)

        return self.alchemical_states

    # TODO run equilibration and run minimisation single state

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
        self._check_alchemical_state_integrity(alchemical_state)

        if not self._U_kl:
            self._iter = 0
            self._U_kl = [[] for _ in range(len(self.alchemical_states))]
            dcd_file_append = False
        else:
            dcd_file_append = True

        if self.write_frame_frequency > 0:
            alchemical_state_id = self.alchemical_states.index(alchemical_state)
            dcd_file = self._create_dcd_file(
                f"{self.output_prefix}_{alchemical_state_id}.dcd",
                alchemical_state.topology,
                alchemical_state.integrator.getStepSize(),
                nsteps,
                append=dcd_file_append,
            )

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

            if (
                self.write_frame_frequency > 0
                and iteration % self.write_frame_frequency == 0
            ):
                dcd_file.writeModel(
                    positions=self._positions, periodicBoxVectors=self._pbc
                )

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

    def set_force_groups(
        self,
        slow_forces: Optional[List[Any]] = None,
        fast_force_group: int = 0,
        slow_force_group: int = 1,
        force_group_dict: Optional[Dict[Any, int]] = None,
    ) -> Dict[Any, int]:
        """
        Set the force groups for the alchemical states.

        Parameters
        ----------
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

    def _get_file_handle(self, filename: str, mode: str = "wb") -> Any:
        """
        Get a file handle.

        Parameters
        ----------
        filename : str
            Name of the file.
        mode : str, default='wb'
            Mode to open the file.

        Returns
        -------
        file_handle : file handle
            File handle.
        """
        return open(filename, mode)

    def _create_dcd_file(
        self, filename: str, topology, dt, interval, append=False, firstStep=0
    ) -> _app.dcdfile.DCDFile:
        """
        Create a DCD file.

        Parameters
        ----------
        filename : str
            Name of the file.
        topology : openmm.app.Topology
            The OpenMM Topology.
        dt : float
            The time step used in the trajectory
        interval : int
            The frequency (measured in time steps) at which states are written to the trajectory
        append : bool, optional, default=False
             If True, open an existing DCD file to append to. If False, create a new file.
        firstStep : int, optional, default=0
            The index of the first step in the trajectory.

        Returns
        -------
        dcdfile : openmm.app.dcdfile.DCDFile
            DCD file.
        """
        mode = "r+b" if append else "w+b"

        if topology is None:
            raise ValueError("Topology must be provided to create a DCD file.")

        logger.info(f"Opening DCD file {filename} in mode {mode}")

        return _app.dcdfile.DCDFile(
            self._get_file_handle(filename, mode=mode),
            topology,
            dt,
            firstStep,
            interval,
            append,
        )
