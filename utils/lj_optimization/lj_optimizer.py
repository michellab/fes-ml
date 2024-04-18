import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class LJOptimizer:
    """Optimize the LJ parameters of a system using OpenMM."""

    def __init__(
        self,
        context: mm.Context,
        topology: app.Topology,
        configurations: Optional[np.ndarray] = None,
        energy_qm: Optional[np.ndarray] = None,
        energy_offset: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the LJ optimization class.

        Parameters
        ----------
        context : openmm.Context
            The OpenMM context object.
        topology : openmm.app.Topology
            The OpenMM topology object.
        configurations : np.ndarray, optional
            The configurations to be used for the optimization.
        energy_qm : np.ndarray, optional
            The QM energy to be used as reference for the optimization.
        energy_offset : np.ndarray, optional
            The reference energy to be used for the optimization.
        verbose : bool, optional
            Whether to print the optimization information.
        """
        if not isinstance(context, mm.Context):
            raise ValueError("Context must be an instance of openmm.Context.")
        if not isinstance(topology, app.Topology):
            raise ValueError("Topology must be an instance of openmm.app.Topology.")

        self._context = context
        self._system = context.getSystem()
        self._topology = topology
        self._configurations = configurations
        self._energy_qm = energy_qm
        self._energy_offset = (
            energy_offset if energy_offset is not None else np.zeros_like(energy_qm)
        )
        self._verbose = verbose

        # Initialize the LJ parameters and atoms to be modified
        self._atoms_opt: Dict[int, List[float]] = None
        self._lj_params: Dict[str, List[float]] = None

    def _update_lj_params_dict(self, opt_params: np.ndarray) -> Dict[str, List[float]]:
        """
        Update the LJ parameters dictionary.

        Parameters
        ----------
        opt_params : np.ndarray
            The optimized LJ parameters.

        Returns
        -------
        dict
            The updated LJ parameters dictionary.
        """
        assert self._lj_params is not None, "No LJ parameters provided."

        # First update the LJ parameters dictionary
        for i, atom_id in enumerate(self._lj_params):
            shift = i * 2  # The shift in the optimized parameters array
            self._lj_params[atom_id][0] = opt_params[shift]
            self._lj_params[atom_id][1] = opt_params[shift + 1]

        return self._lj_params

    def update_lj_params(
        self, opt_params: np.ndarray, force_type: mm.Force
    ) -> mm.System:
        """
        Update the LJ parameters of the atoms in the system.

        Parameters
        ----------
        opt_params : np.ndarray
            The optimized LJ parameters.
        force_type : openmm.Force, optional
            The OpenMM force type containing the LJ parameters.

        Returns
        -------
        openmm.System
            The OpenMM system with the updated LJ parameters.
        """
        # Update the LJ parameters dictionary
        self._lj_params = self._update_lj_params_dict(opt_params)

        # Then update the LJ parameters in the system
        for force in self._system.getForces():
            if isinstance(force, force_type):
                for atom_id, params in self._atoms_opt.items():
                    assert all(
                        isinstance(param, float) for param in params
                    ), "LJ parameters must be floats."

                    if self._verbose:
                        logger.info(
                            f"Setting LJ parameters for atom {atom_id}: {params}"
                        )
                    force.setParticleParameters(int(atom_id), params)

                # Update the context
                force.updateParametersInContext(self._context)

        return self._system

    def get_parameters_particle(self, force_type: mm.Force, p: int) -> List[float]:
        """
        Get the parameters of a particle in a given force instance.

        Parameters
        ----------
        force_type : openmm.Force
            The OpenMM force type.
        p : int
            The particle index.

        Returns
        -------
        list
            The LJ parameters of the particle.
        """
        # Extract the LJ parameters from the system
        for force in self._system.getForces():
            if isinstance(force, force_type):
                if p < force.getNumParticles():
                    return list(force.getParticleParameters(p))
                else:
                    raise ValueError(
                        f"Particle index {p} exceeds the number of particles in the force."
                    )

        raise ValueError(f"Force {force_type} not found in the system.")

    def _generate_identifier(self, res: app.Residue, atom: app.Atom) -> str:
        """
        Generate an identifier for the atom in the residue.

        Parameters
        ----------
        res : openmm.app.Residue
            The residue object.
        atom : openmm.app.Atom
            The atom object.

        Returns
        -------
        str
            The identifier for the atom in the residue.
        """
        return f"{res.name}_{atom.element.symbol}"

    def get_lj_params_opt(
        self,
        residues_atoms: Dict[str, List[str]],
        force_type: mm.Force,
    ) -> OrderedDict:
        """
        Create an ordered dictionary containing the LJ parameters to be optimized.

        Parameters
        ----------
        residues_atoms : Dict[str, List[str]]
            The residues and respective atoms to be optimized.
            This dictionary should have the following format:
            {
                "res1": ["at1", "at2", ...],
                "res2": ["at1", "at2", ...],
                ...
            }
        force_type : openmm.Force
            The OpenMM force containing LJ parameters.

        Returns
        -------
        OrderedDict
            The LJ parameters to be modified.
            This dictionary has the following format:
            {
                "res1_at1": [epsilon, sigma],
                "res1_at2": [epsilon, sigma],
                ...
            }
        """
        # Create a ordered dictionary containing the LJ parameters to be modified
        self._lj_params = OrderedDict()

        # Loop over the residues of the topology, and over the atoms of the residues
        # to find the atoms to be modified
        for res in self._topology.residues():
            if res.name in residues_atoms:
                for atom in res.atoms():
                    if atom.element.symbol not in residues_atoms[res.name]:
                        continue

                    # Create an identifier for the atom in the residue (e.g. "HOH_O")
                    res_at_id = self._generate_identifier(res, atom)

                    if res_at_id not in self._lj_params:
                        # If the LJ parameters are not already in the dictionary
                        # get the parameters of the particle and add them to the dictionary
                        self._lj_params[res_at_id] = self.get_parameters_particle(
                            force_type, atom.index
                        )
                break

        return self._lj_params

    def get_atoms_opt(self, residues_atoms: Dict[str, List[str]]) -> OrderedDict:
        """
        Create an ordered dictionary containing the atoms to be optimized.

        Parameters
        ----------
        residues_atoms : Dict[str, List[str]]
            The residues and respective atoms to be optimized.
            This dictionary should have the following format:
            {
                "res1": ["at1", "at2", ...],
                "res2": ["at1", "at2", ...],
                ...
            }

        Returns
        -------
        OrderedDict
            The atoms to be modified.
        """
        # Create a ordered dictionary containing the LJ parameters of the atoms to be modified
        assert self._lj_params is not None, "No LJ parameters provided."

        self._atoms_opt = OrderedDict()
        for res in self._topology.residues():
            if res.name in residues_atoms:
                for atom in res.atoms():
                    if atom.element.symbol not in residues_atoms[res.name]:
                        continue

                    # Create an identifier for the atom in the residue (e.g. "HOH_O")
                    res_at_id = self._generate_identifier(res, atom)
                    self._atoms_opt[int(atom.id) - 1] = self._lj_params[res_at_id]

        return self._atoms_opt

    def compute_energy(
        self, configurations: np.ndarray, context: Optional[mm.Context] = None
    ) -> np.ndarray:
        """
        Compute the OpenMM energy of a set of configurations.

        Parameters
        ----------
        configurations : np.ndarray
            The configurations to be used for the energy computation.
        context : openmm.Context, optional
            The OpenMM context object.
            Default is None, in which case the context provided during initialization is used.

        Returns
        -------
        np.ndarray
            The energy of the configurations.
        """
        assert configurations is not None, "No configurations provided."

        context = context if context is not None else self._context

        energy = []
        for config_id in range(configurations.shape[2]):
            context.setPositions(configurations[:, :, config_id].T)
            energy.append(
                context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoules_per_mole)
            )

        energy = np.array(energy)

        return energy

    def loss_function(self, params: np.ndarray, force_type: mm.Force) -> float:
        """Compute the loss function for the optimization."""
        assert self._configurations is not None, "No configurations provided."
        assert self._energy_offset is not None, "No reference energy provided."

        logger.info(f"LJ parameters: {params}")
        # Update the LJ parameters
        self.update_lj_params(params, force_type)

        # Compute the energy of the configurations
        energy = self.compute_energy(self._configurations)

        # Subtract the energy offset
        energy -= self._energy_offset

        # Compute the loss function
        squared_diff = (energy - self._energy_qm) ** 2
        loss = np.sum(squared_diff)

        if self._verbose:
            logger.info("-" * 50)
            logger.info(f"{'MM':<16} | {'QM':<16} | {'Offset':<16} | {'Diff':<16}")

            for i in range(len(energy)):
                logger.info(
                    f"{energy[i]:<16.4f} | {self._energy_qm[i]:<16.4f} | {self._energy_offset[i]:<16.4f} | {squared_diff[i]:<16.4f}"
                )

            logger.info(f"Loss function: {loss}")

        return loss

    def run_optimization(
        self,
        residues_atoms: Dict[str, List[str]],
        force_type: mm.Force = mm.CustomNonbondedForce,
        method: str = "Nelder-Mead",
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Run the optimization of the LJ parameters.

        Parameters
        ----------
        residues_atoms : Dict[str, List[str]]
            The residues and respective atoms to be optimized.
            This dictionary should have the following format:
            {
                "res1": ["at1", "at2", ...],
                "res2": ["at1", "at2", ...],
                ...
            }
        force_type : openmm.Force, optional
            The OpenMM force containing LJ parameters.
            Default is CustomNonbondedForce.
        method : str, optional
            The optimization method to be used.
            Default is "Nelder-Mead".

        Returns
        -------
        np.ndarray
            The optimized LJ parameters.
        """
        # Get the LJ parameters to be modified
        self.get_lj_params_opt(residues_atoms, force_type)
        self.get_atoms_opt(residues_atoms)

        # Define the initial guess
        x0 = [param for params in self._lj_params.values() for param in params]

        # Optimize the LJ parameters
        res = minimize(
            self.loss_function, x0, method=method, args=(force_type), *args, **kwargs
        )

        logger.info(res)
        logger.info(f"Optimization result: {res.x}")

        return res.x

    def write_optimized_parameters(self, filename: str) -> None:
        """
        Write the optimized LJ parameters to a file.

        Parameters
        ----------
        filename : str
            The name of the file to write the optimized parameters.
        """
        with open(filename, "w") as f:
            for atom_id, params in self._lj_params.items():
                format_string = f"{atom_id}"
                for param in params:
                    format_string += f" {param:.6f}"
                format_string += "\n"
                f.write(format_string)

    @staticmethod
    def read_parameters(file: str) -> Dict[str, List[float]]:
        """
        Read the parameters from the file.

        Parameters
        ----------
        file : str
            The file to read.

        Returns
        -------
        Dict[str, List[float]]
            The parameters as a dictionary with the following format:
            {
                "res1_at1": [sigma, epsilon],
                "res1_at2": [sigma, epsilon],
                ...
            }
        """
        with open(file, "r") as f:
            lines = f.readlines()
            parameters = {}
            for line in lines:
                splitted_line = line.split()
                key = splitted_line[0]
                params = [float(i) for i in splitted_line[1:]]

                parameters[key] = params

            return parameters

    def update_system_parameters_from_file(
        self,
        params_file: str,
        force_type: Optional[mm.Force] = mm.CustomNonbondedForce,
    ) -> Tuple[mm.System, mm.Context]:
        """
        Update the parameters of the system with the parameters in the file.
        The context is also updated with the new parameters.

        Parameters
        ----------
        params_file : str
            The file containing the parameters.
        force_type : openmm.Force, optional
            The force type to update, by default mm.CustomNonbondedForce

        Returns
        -------
        openmm.System, openmm.Context
            The updated system and context.
        """
        logger.info(f"Updating the system parameters from file: {params_file}")
        # Read the parameters from the file
        opt_params = self.read_parameters(params_file)

        # Get the atoms to modify in the system
        atoms_mod = []
        for key in opt_params.keys():
            res_name, atom_name = key.split("_")
            for res in self._topology.residues():
                if res.name == res_name:
                    for atom in res.atoms():
                        if atom.element.symbol == atom_name:
                            atoms_mod.append(atom.index)

            # Update the LJ parameters in the system
            for force in self._system.getForces():
                if isinstance(force, force_type):
                    for atom_id in atoms_mod:
                        logger.info(
                            f"Setting LJ parameters for atom {atom_id}: {opt_params[key]}"
                        )
                        force.setParticleParameters(atom_id, opt_params[key])

                    force.updateParametersInContext(self._context)

        return self._system, self._context
