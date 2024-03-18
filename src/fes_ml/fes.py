import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import openmm as _mm
import openmm.unit as _unit

from .alchemical import AlchemicalState
from .alchemical import alchemical_factory as alchemical_factory


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
    alchemical_atoms : list of int
        List of atom indices to be alchemically modified.
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
    """

    __serializables__ = [
        "crd_file",
        "top_file",
        "lambda_schedule",
        "alchemical_atoms",
        "_iter",
        "_alc_id",
        "_positions",
        "_pbc",
        "_U_kl",
        "_U_kln",
        "_create_alchemical_states_args",
        "_create_alchemical_states_kwargs",
    ]

    _LAMBDA_PARAMS = ["lambda_lj", "lambda_q", "lambda_interpolate", "lambda_emle"]

    def __init__(
        self,
        crd_file: Optional[str] = None,
        top_file: Optional[str] = None,
        lambda_schedule: Optional[dict] = None,
        alchemical_atoms: Optional[List[int]] = None,
        checkpoint_frequency: int = 100,
        checkpoint_file: str = "checkpoint.pickle",
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
        lambda_schedule : dict, optional, default=None
            Dictionary with the lambda values for the alchemical states.
            The keys of the dictionary are the lambda parameters and the values are lists of lambda values.
            Available lambda parameters are: "lambda_lj", "lambda_q", "lambda_interpolate", "lambda_emle".
        alchemical_atoms : list of int, optional, default=None
            List of atom indices to be alchemically modified.
        checkpoint_frequency : int, optional, default=100
            Frequency to save the state of the object.
        checkpoint_file : str, optional, default="checkpoint.pickle"
            Path to the checkpoint file.
        restart : bool, optional, default=False
            Whether to restart from the last checkpoint.
        """
        self.crd_file = crd_file
        self.top_file = top_file
        self.lambda_schedule = lambda_schedule
        self.alchemical_atoms = alchemical_atoms
        self.alchemical_states: List[AlchemicalState] = None
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_file = checkpoint_file

        # Checkpoint variables
        self._iter: Optional[int] = None
        self._alc_id: Optional[int] = None
        self._pos: Optional[Any] = None
        self._pbc: Optional[Any] = None
        self._U_kl: Optional[List[List[float]]] = None
        self._U_kln: Optional[List[List[List[float]]]] = None
        self._create_alchemical_states_args: Optional[Tuple[Any, ...]] = None
        self._create_alchemical_states_kwargs: Optional[Dict[str, Any]] = None
        self._force_groups: Optional[Dict[_mm.Force, int]] = None

        if restart:
            assert os.path.exists(
                self.checkpoint_file
            ), f"Checkpoint file {self.checkpoint_file} does not exist."
            print(f"Loading state from {self.checkpoint_file}")
            with open(self.checkpoint_file, "rb") as f:
                state = pickle.load(f)
            self.__setstate__(state)

    def __getstate__(self) -> Dict[str, Any]:
        """Get the state of the object."""
        state = {key: getattr(self, key) for key in self.__serializables__}
        return state

    def __setstate__(self, state) -> None:
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
        with open("checkpoint.pickle", "wb") as f:
            pickle.dump(state, f)

        print(f"Saved state to {self.checkpoint_file}")

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
        # Store the arguments and keyword arguments to recreate the alchemical states
        # when the object is deserialized
        self._create_alchemical_states_args = args
        self._create_alchemical_states_kwargs = kwargs
        self.lambda_schedule = lambda_schedule
        self.alchemical_atoms = alchemical_atoms

        # Assert all keys in lambda_schedule are valid
        for key in lambda_schedule:
            assert (
                key in self._LAMBDA_PARAMS
            ), f"Invalid lambda parameter {key} in `lambda_schedule`. Valid lambdas are {self._LAMBDA_PARAMS}."

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

        self.alchemical_states = []

        for i in range(nstates):
            alchemical_state = alchemical_factory.create_alchemical_state(
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

    def run_minimization_batch(
        self,
        tolerance: _unit.Quantity = 10 * _unit.kilojoules_per_mole / _unit.nanometer,
        max_iterations: int = 0,
        reporter=None,
    ) -> List[AlchemicalState]:
        """
        Run a batch of minimizations.

        Parameters
        ----------
        tolerance : openmm.unit.Quantity
            The energy tolerance to which the system should be minimized.
        max_iterations : int, optional, default=0
            The maximum number of iterations to run the minimization.
        reporter : object, optional, default=None
            A reporter object to report the minimization progress.

        Returns
        -------
        alchemical_states : list of AlchemicalState
            List of alchemical states.
        """
        assert (
            self.alchemical_states is not None
        ), "The alchemical states have not been created. Run `create_alchemical_states` first."

        for alc in self.alchemical_states:
            print(f"Minimizing {alc}")
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
            print(f"Equilibrating {alc}")
            self._check_alchemical_state_integrity(alc)
            alc.integrator.step(nsteps)

        return self.alchemical_states

    def run_single_state(
        self,
        niterations: int,
        nsteps: int,
        alchemical_state: AlchemicalState = None,
        window: int = None,
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

        # Resume from the last checkpoint
        if not self._U_kl:
            self._iter = 0
            self._U_kl = [[] for _ in range(len(self.alchemical_states))]

        print(
            f"Starting {alchemical_state} from iteration {self._iter} / {niterations}"
        )

        kT = (
            _unit.AVOGADRO_CONSTANT_NA
            * _unit.BOLTZMANN_CONSTANT_kB
            * alchemical_state.integrator.getTemperature()
        )

        for iteration in range(self._iter, niterations):
            print(f"{alchemical_state} iteration {iteration} / {niterations}")

            alchemical_state.integrator.step(nsteps)
            self._positions = alchemical_state.context.getState(
                getPositions=True
            ).getPositions()
            self._pbc = alchemical_state.context.getState().getPeriodicBoxVectors()

            # Compute energies at all alchemical states
            for l, alc_ in enumerate(self.alchemical_states):
                alc_.context.setPositions(self._positions)
                alc_.context.setPeriodicBoxVectors(*self._pbc)
                self._U_kl[l].append(
                    alc_.context.getState(getEnergy=True).getPotentialEnergy() / kT
                )

            # Save the checkpoint
            if iteration % self.checkpoint_frequency == 0 and iteration > 0:
                self._iter = iteration + 1
                self._save_state()

        return self._U_kl

    def run_production_batch(self, niterations, nsteps) -> List[List[List[float]]]:
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
        assert (
            self.alchemical_states is not None
        ), "The alchemical states have not been created. Run `create_alchemical_states` first."

        # Resume from the last checkpoint
        if not self._U_kln:
            self._alc_id = 0
            self._U_kln = []

        print(
            f"Starting production run with {self.alchemical_states[self._alc_id]} with id {self._alc_id}"
        )
        for alc_id in range(self._alc_id, len(self.alchemical_states)):
            alc = self.alchemical_states[alc_id]
            self._U_kln.append(self.run_single_state(niterations, nsteps, alc))
            self._alc_id = alc_id + 1
            self._iter = 0
            self._save_state()

        return self._U_kln

    def set_force_groups(
        self,
        slow_forces: Optional[List[_mm.Force]] = None,
        fast_force_group: int = 0,
        slow_force_group: int = 1,
        force_group_dict: Optional[Dict[_mm.Force, int]] = None,
    ) -> None:
        """
        Set the force groups for the alchemical states.

        Parameters
        ----------
        slow_forces : list of openmm.Force
            List of forces to be considered slow.
        fast_force_group : int
            Force group for the fast forces.
        slow_force_group : int, optional, default=1
            Force group for the slow forces.
        force_group_dict : dict, optional, default=None
            Dictionary with the force group for each force.
            If provided, it takes precedence over the other arguments and the force groups are set according to the dictionary.
            Otherwise, the force groups are set according to the other arguments.

        Notes
        -----
        The force groups are set as follows:
        - If `force_group_dict` is provided, the force groups are set according to the dictionary.
        - If `slow_forces` is provided, the forces in `slow_forces` are set to `slow_force_group` and the others to `fast_force_group`.
        - If `slow_forces` is not provided, all forces are set to `fast_force_group`.
        """
        if force_group_dict is not None:
            assert all(
                [isinstance(force, _mm.Force) for force in force_group_dict]
            ), "All keys in `force_group_dict` must be instances of `openmm.Force`."
            for state in self.alchemical_states:
                for force in state.system.getForces():
                    force.setForceGroup(force_group_dict[force])

            state.context.reinitialize(preserveState=True)
        else:
            if slow_forces is not None:
                assert all(
                    [isinstance(force, _mm.Force) for force in slow_forces]
                ), "All forces in `slow_forces` must be instances of `openmm.Force`."
                for state in self.alchemical_states:
                    for force in state.system.getForces():
                        if force in slow_forces:
                            force.setForceGroup(slow_force_group)
                        else:
                            force.setForceGroup(fast_force_group)

                    state.context.reinitialize(preserveState=True)
            else:
                for state in self.alchemical_states:
                    for force in state.system.getForces():
                        force.setForceGroup(fast_force_group)

                    state.context.reinitialize(preserveState=True)

        # Store the force groups
        self._force_groups = {
            force: force.getForceGroup()
            for state in self.alchemical_states
            for force in state.system.getForces()
        }
