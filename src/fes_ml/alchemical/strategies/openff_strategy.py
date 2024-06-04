"""OpenFF alchemical state creation strategy."""

import logging
import os as _os
import shutil as _shutil
from copy import deepcopy as _deepcopy
from typing import Any, Dict, List, Optional, Union

# OpenMM imports
import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit
from openff.interchange.components._packmol import UNIT_CUBE as _UNIT_CUBE
from openff.interchange.components._packmol import pack_box as _pack_box
from openff.interchange.components.mdconfig import MDConfig as _MDConfig
from openff.interchange.interop.openmm._positions import (
    to_openmm_positions as _to_openmm_positions,
)

# OpenFF imports
from openff.toolkit import Molecule as _Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField as _ForceField
from openff.units import unit as _offunit

# FES-ML imports
from ..alchemical_state import AlchemicalState
from .base_strategy import AlchemicalStateCreationStrategy

logger = logging.getLogger(__name__)


class OpenFFCreationStrategy(AlchemicalStateCreationStrategy):
    """Strategy for creating alchemical states using OpenFF."""

    _TMP_DIR = "tmp_fes_ml_openmm"

    _MDCONFIG_DICT = {
        "periodic": True,
        "constraints": "h-bonds",
        "vdw_method": "cutoff",
        "vdw_cutoff": _offunit.Quantity(12.0, "angstrom"),
        "mixing_rule": "lorentz-berthelot",
        "switching_function": True,
        "switching_distance": _offunit.Quantity(11.0, "angstrom"),
        "coul_method": "pme",
        "coul_cutoff": _offunit.Quantity(12.0, "angstrom"),
    }

    _MDCONFIG_DICT_VACUUM = {
        "periodic": False,
        "constraints": "h-bonds",
        "vdw_method": "no-cutoff",
        "mixing_rule": "lorentz-berthelot",
        "switching_function": True,
        "coul_method": "no-cutoff",
    }

    _PACKMOL_KWARGS = {
        "box_shape": _UNIT_CUBE,
        "mass_density": 1.0 * _offunit.gram / _offunit.milliliter,
    }

    _N_MOLECULES = {
        "ligand": 1,
        "solvent": 1000,
        "protein": 1,
    }

    _DEFAULT_FORCEFIELDS = ["openff-2.0.0.offxml", "tip3p.offxml"]

    @staticmethod
    def _create_integrator(
        temperature: Union[float, _unit.Quantity],
        friction: Union[float, _unit.Quantity],
        timestep: Union[float, _unit.Quantity],
    ) -> _mm.Integrator:
        """
        Create an OpenMM integrator.

        Parameters
        ----------
        temperature : float or _unit.Quantity
            The temperature in Kelvin.
        friction : float or _unit.Quantity
            The friction coefficient in 1/ps.
        timestep : float or _unit.Quantity
            The timestep in ps.

        Returns
        -------
        _mm.Integrator
            The OpenMM integrator.
        """
        if temperature is not None:
            if isinstance(temperature, _unit.Quantity):
                temperature = temperature.value_in_unit(_unit.kelvin)
            elif not isinstance(temperature, float):
                raise ValueError("Temperature must be a float or a Quantity.")

            if isinstance(friction, _unit.Quantity):
                friction = friction.value_in_unit(_unit.picosecond**-1)
            elif not isinstance(friction, float):
                raise ValueError("Friction must be a float or a Quantity.")

            if isinstance(timestep, _unit.Quantity):
                timestep = timestep.value_in_unit(_unit.picosecond)
            elif not isinstance(timestep, float):
                raise ValueError("Timestep must be a float or a Quantity.")

            return _mm.LangevinIntegrator(temperature, friction, timestep)
        else:
            return _mm.VerletIntegrator(timestep)

    @staticmethod
    def _create_ligand_molecule(sdf_file_ligand: str, smiles_ligand: str) -> _Molecule:
        """
        Create a ligand molecule from an SDF file or a SMARTS pattern.

        Parameters
        ----------
        sdf_file_ligand : str
            The path to the SDF file containing the ligand.
        smiles_ligand : str
            The SMARTS pattern for the ligand.

        Returns
        -------
        _Molecule
            The ligand molecule.
        """
        if sdf_file_ligand is not None:
            ligand = _Molecule.from_file(sdf_file_ligand)
        elif smiles_ligand is not None:
            ligand = _Molecule.from_smiles(smiles_ligand)
        else:
            raise ValueError(
                "Please provide either an SDF file or a SMARTS pattern for the ligand."
            )
        return ligand

    @staticmethod
    def _create_solvent_molecule(
        sdf_file_solvent: str, smiles_solvent: str
    ) -> Union[_Molecule, None]:
        """
        Create a solvent molecule from an SDF file or a SMARTS pattern.

        Parameters
        ----------
        sdf_file_solvent : str
            The path to the SDF file containing the solvent.
        smiles_solvent : str
            The SMARTS pattern for the solvent.

        Returns
        -------
        _Molecule or None
            The solvent molecule.
        """
        if sdf_file_solvent is not None:
            solvent = _Molecule.from_file(sdf_file_solvent)
        elif smiles_solvent is not None:
            solvent = _Molecule.from_smiles(smiles_solvent)
        else:
            solvent = None
            logger.debug("No solvent provided. Assuming the ligand is in vacuum.")

        return solvent

    @staticmethod
    def _create_protein_molecule() -> Union[_Molecule, None]:
        return None

    @staticmethod
    def _solvate(
        molecules: Dict[str, Union[_Molecule, None]],
        packmol_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Solvate the system using Packmol.

        Parameters
        ----------
        molecules : list of _Molecule
            The list of molecules to solvate.
        packmol_kwargs : dict, optional, default=None
            A dictionary of keyword arguments for the Packmol pack_box function.

        Returns
        -------
        _Topology
            The solvated (or not) topology.
        """
        if molecules["solvent"] is None:
            logger.debug("Not solvating the system.")
            topology_off = molecules["ligand"].to_topology()
        else:
            logger.debug("Solvating the system.")

            mols = [mol for _, mol in molecules.items() if mol is not None]

            if packmol_kwargs is None:
                packmol_kwargs = _deepcopy(OpenFFCreationStrategy._PACKMOL_KWARGS)

            if "number_of_copies" not in packmol_kwargs:
                number_of_copies = [
                    OpenFFCreationStrategy._N_MOLECULES[mol_name]
                    for mol_name, mol in molecules.items()
                    if mol is not None
                ]

            topology_off = _pack_box(
                molecules=mols,
                number_of_copies=number_of_copies,
                **packmol_kwargs,
            )

        return topology_off

    @staticmethod
    def _get_alchemical_atoms(
        topology: _app.Topology, alchemical_atoms: Optional[List[int]]
    ) -> List[int]:
        """
        Get the alchemical atoms.

        Parameters
        ----------
        topology : _app.Topology
            The OpenMM topology.
        alchemical_atoms : list of int
            A list of atom indices to be alchemically modified.

        Returns
        -------
        list of int
            The alchemical atoms.
        """
        if alchemical_atoms is None:
            logger.debug(
                "No alchemical atoms provided. Assuming the whole ligand is alchemical."
            )
            alchemical_atoms = [
                atom.index for atom in list(topology.chains())[0].atoms()
            ]
            if len(alchemical_atoms) > 100:
                logger.warning(
                    f"The size of the alchemical region is {len(alchemical_atoms)}. This seems like too much."
                )

        else:
            assert isinstance(
                alchemical_atoms, list
            ), "Alchemical_atoms must be a list of int."

        logger.debug(f"Alchemical atoms: {alchemical_atoms}")
        return alchemical_atoms

    @staticmethod
    def _create_barostat(
        pressure: Union[float, _unit.Quantity, None],
        temperature: Union[float, _unit.Quantity],
        frequency: int = 25,
    ) -> Union[_mm.MonteCarloBarostat, None]:
        """
        Create a Monte Carlo Barostat.

        Parameters
        ----------
        pressure : float or _unit.Quantity or None
            The pressure in bar.
        temperature : float or _unit.Quantity
            The temperature in Kelvin.
        frequency : int, optional, default=25
            The frequency at which MC pressure changes should be attempted (in time steps).

        Returns
        -------
        _mm.MonteCarloBarostat or None
            The Monte Carlo Barostat.
        """
        if pressure is not None:
            if isinstance(pressure, _unit.Quantity):
                pressure = pressure.value_in_unit(_unit.bar)
            elif not isinstance(pressure, float):
                raise ValueError("Pressure must be a float or a Quantity.")

            if isinstance(temperature, _unit.Quantity):
                temperature = temperature.value_in_unit(_unit.kelvin)
            elif not isinstance(temperature, float):
                raise ValueError("Temperature must be a float or a Quantity.")

            return _mm.MonteCarloBarostat(pressure, temperature, frequency)
        return None

    def create_alchemical_state(
        self,
        alchemical_atoms: List[int],
        lambda_schedule: Dict[str, Union[float, int]],
        sdf_file_ligand: Optional[str] = None,
        sdf_file_solvent: Optional[str] = None,
        smiles_ligand: Optional[str] = None,
        smiles_solvent: Optional[str] = None,
        temperature: Union[float, _unit.Quantity] = 298.15 * _unit.kelvin,
        pressure: Union[float, _unit.Quantity, None] = 1.0 * _unit.atmosphere,
        mdconfig_dict: Optional[Dict[str, Any]] = None,
        packmol_kwargs: Optional[Dict[str, Any]] = None,
        hydrogen_mass: Optional[Union[float, _unit.Quantity]] = 1.007947 * _unit.amu,
        forcefields: Optional[List[str]] = None,
        remove_constraints: bool = True,
        integrator: Optional[Any] = None,
        friction: Union[float, _unit.Quantity] = 1.0 / _unit.picosecond,
        timestep: Union[float, _unit.Quantity] = 1.0 * _unit.femtosecond,
        write_pdb: bool = True,
        keep_tmp_files: bool = True,
        modifications_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> AlchemicalState:
        """
        Create an alchemical state for the given λ values using OpenMM Systems created with Sire.

        Parameters
        ----------
        alchemical_atoms : list of int
            A list of atom indices to be alchemically modified.
        lambda_schedule : dict
            A dictionary mapping the name of the alchemical modification to the λ value.
        sdf_file_ligand : str, optional, default=None
            The path to the SDF file containing the ligand.
        sdf_file_solvent : str, optional, default=None
            The path to the SDF file containing a solvent molecule.
        smiles_ligand : str, optional, default=None
            The SMARTS pattern for the ligand.
        smiles_solvent : str, optional
            The SMARTS pattern for the solvent. If None, the ligand is assumed to be in vacuum.
            For water use '[H:2][O:1][H:3]'.
        temperature : float or _unit.Quantity, optional, default=298.15
            The temperature in Kelvin.
        mdconfig_dict
            A dictionary of keyword arguments for the MDConfig object.
            See: https://docs.openforcefield.org/projects/interchange/en/stable/_autosummary/openff.interchange.components.mdconfig.MDConfig.html#openff.interchange.components.mdconfig.MDConfig
        packmol_kwargs : dict, optional, default=None
            A dictionary of keyword arguments for the Packmol pack_box function.
            The list of molecules is already passed and should not be provided.
            An attempt is made to infer the number of molecules from the number of copies if not provided.
            See: TODO: add docs for packmol_kwargs
        hydrogen_mass : float or _unit.Quantity, optional, default=None
            The mass of the hydrogen atom. If None, the mass is not changed.
        forcefields : list of str, optional, default=None
            The list of forcefields to use. If None, those in _DEFAULT_FORCEFIELDS are used.
        remove_constraints : bool, optional, default=True
            Whether to remove constraints involving the alchemical atoms.
        pressure : float or _unit.Quantity or None, optional, default=1.0
            The pressure in bar. If None, the Monte Carlo Barostat is not added.
        integrator : Any, optional, default=None
            The OpenMM integrator to use. If None, the default is a LangevinMiddle integrator with a 1 fs
            timestep and a 298.15 K temperature (if temperature is not None). Otherwise, a Verlet integrator
            with the provided timestep is used.
        friction : float or _unit.Quantity, optional, default=1.0 / _unit.picosecond
            The friction coefficient in 1/ps. Only necessary for Langevin-type integrators.
        timestep : float or _unit.Quantity, optional, default=1.0 * _unit.femtosecond
            The timestep in ps of the integrator.
        write_pdb : bool, optional, default=True
            Save coordinates and topology to a PDB file.
        keep_tmp_files : bool, optional, default=True
            Whether to keep the temporary files created by the strategy.
        modifications_kwargs : dict
            A dictionary of keyword arguments for the modifications.

        Returns
        -------
        AlchemicalState
            The alchemical state.
        """
        logger.debug("")
        logger.debug("=" * 100)

        # Generate local copies of the system generator kwargs
        if any([sdf_file_solvent, smiles_solvent]):
            mdconfig_dict = (
                _deepcopy(self._MDCONFIG_DICT)
                if mdconfig_dict is None
                else _deepcopy(mdconfig_dict)
            )
        else:
            mdconfig_dict = (
                _deepcopy(self._MDCONFIG_DICT_VACUUM)
                if mdconfig_dict is None
                else _deepcopy(mdconfig_dict)
            )

        passed_args = locals()
        passed_args["mdconfig_kwargs"] = mdconfig_dict

        # Report the creation settings
        self._report_dict(passed_args, dict_name="OpenFF creation settings")

        # Create temporary directory if it does not exist
        _os.makedirs(self._TMP_DIR, exist_ok=True)

        # Create local variables
        additional_forces: List[Union[_mm.Force, None]] = []
        molecules: Dict[str, Union[_Molecule, None]] = {}

        # Create the ligand, protein, and solvent molecules
        ligand = self._create_ligand_molecule(sdf_file_ligand, smiles_ligand)
        protein = self._create_protein_molecule()
        solvent = self._create_solvent_molecule(sdf_file_solvent, smiles_solvent)

        # Set the molecules
        molecules["ligand"] = ligand
        molecules["protein"] = protein
        molecules["solvent"] = solvent

        # Generate conformers for the ligand
        ligand.generate_conformers()

        # Solvate the system
        topology_off = self._solvate(molecules, packmol_kwargs)

        # Convert topology to OpenMM
        topology = topology_off.to_openmm()

        # Create the Interchange object
        ffs = forcefields or self._DEFAULT_FORCEFIELDS
        interchange = _ForceField(*ffs).create_interchange(topology_off)

        # Apply the MDConfig settings to the Interchange object
        mdconfig = _MDConfig()
        for key, value in mdconfig_dict.items():
            setattr(mdconfig, key, value)
        mdconfig.apply(interchange)

        self._report_dict(
            {attr: getattr(mdconfig, attr) for attr in vars(mdconfig)},
            dict_name="MDConfig settings",
        )

        # Create the simulation from the Interchange object
        if isinstance(hydrogen_mass, _unit.Quantity):
            hmr = hydrogen_mass.value_in_unit(_unit.amu)
        elif isinstance(hydrogen_mass, float):
            hmr = hydrogen_mass
        else:
            raise ValueError("Hydrogen mass must be a float or a Quantity.")

        system = interchange.to_openmm_system(
            combine_nonbonded_forces=True,
            add_constrained_forces=True,
            hydrogen_mass=hmr,
        )

        # Create barostat (only if system is periodic)
        if (
            topology.getPeriodicBoxVectors() is not None
        ) or system.usesPeriodicBoundaryConditions():
            additional_forces.append(self._create_barostat(pressure, temperature))

        # Add additional forces
        for force in additional_forces:
            if force is not None:
                system.addForce(force)

        # Infer the alchemical atoms if not provided
        alchemical_atoms = self._get_alchemical_atoms(topology, alchemical_atoms)

        # Report the energy decomposition before applying the alchemical modifications
        positions = _to_openmm_positions(interchange, include_virtual_sites=True)
        tmp_context = _mm.Context(system, _mm.VerletIntegrator(1))
        tmp_context.setPositions(positions)

        if int(_os.getenv("FES_ML_LOG_DEVEL", False)):
            self._report_energy_decomposition(tmp_context, system)

        # Remove constraints involving the alchemical atoms
        if remove_constraints:
            system = self._remove_constraints(system, alchemical_atoms)

        # Create/update the modifications kwargs
        modifications_kwargs = _deepcopy(modifications_kwargs) or {}
        if any(key in lambda_schedule for key in ["EMLEPotential"]):
            raise NotImplementedError(
                "The EMLEPotential is not yet supported in the OpenFF strategy."
            )

        if any(key in lambda_schedule for key in ["EMLEPotential", "MLInterpolation"]):
            modifications_kwargs["EMLEPotential"] = modifications_kwargs.get(
                "EMLEPotential", {}
            )
            # TODO: make interchage write parm7 for EMLE
            # modifications_kwargs["EMLEPotential"]["mols"] = mols
            # modifications_kwargs["EMLEPotential"]["parm7"] = alchemical_prm7[0]
            # modifications_kwargs["EMLEPotential"]["mm_charges"] = _np.asarray(
            #    [atom.charge().value() for atom in mols.atoms(alchemical_atoms)]
            # )
        if any(
            key in lambda_schedule
            for key in ["MLPotential", "MLInterpolation", "MLCorrection"]
        ):
            modifications_kwargs["MLPotential"] = modifications_kwargs.get(
                "MLPotential", {}
            )
            modifications_kwargs["MLPotential"]["topology"] = topology

        # Run the Alchemist
        self._run_alchemist(
            system,
            alchemical_atoms,
            lambda_schedule,
            modifications_kwargs=modifications_kwargs,
        )

        # Create the integrator
        if integrator is None:
            integrator = self._create_integrator(temperature, friction, timestep)
        else:
            assert isinstance(
                integrator, _mm.Integrator
            ), "integrator must be an OpenMM Integrator."
            integrator = _deepcopy(integrator)

        # Create the simulation
        simulation = _app.Simulation(
            topology=interchange.to_openmm_topology(),
            system=system,
            integrator=integrator,
        )

        # Set the positions
        simulation.context.setPositions(positions)

        if int(_os.getenv("FES_ML_LOG_DEVEL", False)):
            # Report the energy decomposition after applying the alchemical modifications
            # Expensive!
            self._report_energy_decomposition(simulation.context, simulation.system)

        # Create the AlchemicalState
        alc_state = AlchemicalState(
            system=simulation.system,
            context=simulation.context,
            integrator=simulation.integrator,
            simulation=simulation,
            topology=topology,
            modifications=lambda_schedule,
        )

        # Save the topology and positions to a PDB file
        if write_pdb:
            topology_off.to_file(
                _os.path.join(self._TMP_DIR, "topology.pdb"),
                _to_openmm_positions(interchange, include_virtual_sites=False),
            )

        # Clean up the temporary directory
        if not keep_tmp_files:
            _shutil.rmtree(self._TMP_DIR)

        logger.debug("Alchemical state created successfully.")
        logger.debug("=" * 100)

        return alc_state
