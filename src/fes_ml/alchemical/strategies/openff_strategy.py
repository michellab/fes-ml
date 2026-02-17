"""OpenFF alchemical state creation strategy."""

import logging
import os as _os
import re as _re
import shutil as _shutil
from copy import deepcopy as _deepcopy
from typing import Any, Dict, List, Optional, Union

# OpenMM imports
import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit

# OpenFF imports
from openff.interchange.components._packmol import UNIT_CUBE as _UNIT_CUBE
from openff.interchange.components._packmol import pack_box as _pack_box
from openff.interchange.components.mdconfig import MDConfig as _MDConfig
from openff.interchange.exceptions import (
    UnsupportedExportError as _UnsupportedExportError,
)
from openff.interchange.interop.openmm._positions import (
    to_openmm_positions as _to_openmm_positions,
)
from openff.toolkit import Molecule as _Molecule
from openff.toolkit import Topology as _Topology
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
        "target_density": 1.0 * _offunit.gram / _offunit.milliliter,
    }

    _N_MOLECULES = {
        "ligand": 1,
        "solvent": 1000,
        "protein": 1,
    }

    _DEFAULT_FORCEFIELDS = ["openff_unconstrained-2.0.0.offxml", "tip3p.offxml"]

    _OFF_TO_OMM_MAPPING = {
        # Constraints
        "h-bonds": _app.HBonds,
        "all-bonds": _app.AllBonds,
        "all-angles": _app.HAngles,
        # Nonbonded methods
        "no-cutoff": _app.NoCutoff,
        "pme": _app.PME,
        "cutoff": _app.CutoffPeriodic,
    }

    @staticmethod
    def is_mapped_smiles(smiles: str) -> bool:
        """
        Check if the given SMILES string is a mapped SMILES.

        Parameters
        ----------
        smiles
            The smiles to check.

        Returns
        -------
        bool
            Whether the smiles is mapped or not.
        """
        # Regular expression to find atom mapping numbers (e.g., :1, :2, etc.)
        pattern = _re.compile(r":[0-9]+")

        # Search for the pattern in the SMILES string
        return bool(pattern.search(smiles))

    @staticmethod
    def _apply_hmr(interchange: Any, system: _mm.System, hydrogen_mass: _unit.Quantity) -> _mm.System:
        """
        Apply hydrogen mass repartitioning to the system.

        Parameters
        ----------
        interchange : openff.interchange.Interchange
            The Interchange object.
        system : openmm.System
            The OpenMM System.
        hmr : float
            The mass of the hydrogen atom.

        Returns
        -------
        openmm.System
            The OpenMM System.

        Notes
        -----
        This method assumes that the water molecule is rigid and that the virtual sites are not involved in the HMR.
        Code adapted from https://github.com/openforcefield/openff-interchange/blob/426e3ebc630604b2f15fab014410fac0e48aa514/openff/interchange/interop/openmm/__init__.py#L173-L226.
        """
        logger.warning("Applying hydrogen mass repartitioning on a system with virtual sites!")
        logger.warning("Assuming the water molecule is rigid.")
        water = _Molecule.from_smiles("O")

        def _is_water(molecule: _Molecule) -> bool:
            return molecule.is_isomorphic_with(water)

        for bond in interchange.topology.bonds:
            heavy_atom, hydrogen_atom = bond.atoms
            if heavy_atom.atomic_number == 1:
                heavy_atom, hydrogen_atom = hydrogen_atom, heavy_atom
            if (
                (hydrogen_atom.atomic_number == 1)
                and (heavy_atom.atomic_number != 1)  # noqa: W503
                and not (_is_water(hydrogen_atom.molecule))  # noqa: W503
            ):
                hydrogen_index = interchange.topology.atom_index(hydrogen_atom)
                heavy_index = interchange.topology.atom_index(heavy_atom)

                # This will need to be wired up through the OpenFF-OpenMM particle index map
                # when virtual sites + HMR are supported
                mass_to_transfer = hydrogen_mass - system.getParticleMass(hydrogen_index)

                system.setParticleMass(
                    hydrogen_index,
                    hydrogen_mass,
                )

                system.setParticleMass(
                    heavy_index,
                    system.getParticleMass(heavy_index) - mass_to_transfer,
                )

        return system

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
            if OpenFFCreationStrategy.is_mapped_smiles(smiles_ligand):
                ligand = _Molecule.from_mapped_smiles(smiles_ligand)
            else:
                ligand = _Molecule.from_smiles(smiles_ligand)
        else:
            raise ValueError("Please provide either an SDF file or a SMARTS pattern for the ligand.")
        return ligand

    @staticmethod
    def _create_solvent_molecule(sdf_file_solvent: str, smiles_solvent: str) -> Union[_Molecule, None]:
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
            if OpenFFCreationStrategy.is_mapped_smiles(smiles_solvent):
                solvent = _Molecule.from_mapped_smiles(smiles_solvent)
            else:
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
                packmol_kwargs_local = _deepcopy(OpenFFCreationStrategy._PACKMOL_KWARGS)
            else:
                # Update the packmol_kwargs with the default values
                # Values in packmol_kwargs take precedence
                packmol_kwargs_local = {
                    **packmol_kwargs,
                    **_deepcopy(OpenFFCreationStrategy._PACKMOL_KWARGS),
                }

            if "number_of_copies" not in packmol_kwargs_local:
                number_of_copies = [OpenFFCreationStrategy._N_MOLECULES[mol_name] for mol_name, mol in molecules.items() if mol is not None]
            else:
                number_of_copies = [
                    packmol_kwargs_local["number_of_copies"][mol_name] for mol_name, mol in molecules.items() if mol is not None
                ]

                packmol_kwargs_local.pop("number_of_copies")

            if number_of_copies[1] == 1:
                topology_off = _Topology.from_molecules(mols)
            else:
                topology_off = _pack_box(
                    molecules=mols,
                    number_of_copies=number_of_copies,
                    **packmol_kwargs_local,
                )

        return topology_off

    @staticmethod
    def _get_alchemical_atoms(topology: _app.Topology, alchemical_atoms: Optional[List[int]]) -> List[int]:
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
            logger.debug("No alchemical atoms provided. Assuming the whole ligand is alchemical.")
            alchemical_atoms = [atom.index for atom in list(topology.chains())[0].atoms()]
            if len(alchemical_atoms) > 100:
                logger.warning(f"The size of the alchemical region is {len(alchemical_atoms)}. This seems like too much.")

        else:
            assert isinstance(alchemical_atoms, list), "Alchemical_atoms must be a list of int."

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

    @staticmethod
    def _get_openmm_charges(system) -> List[float]:
        """
        Get the original charges of the OpenMM system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System.

        Returns
        -------
        list : float
            The charges of the system.
        """
        nb_forces = [force for force in system.getForces() if isinstance(force, _mm.NonbondedForce)]
        if len(nb_forces) > 1:
            raise ValueError("The system must not contain more than one NonbondedForce.")
        elif len(nb_forces) == 0:
            logger.warning("The system does not contain a NonbondedForce and therefore no charge scaling will be applied.")
            return system
        else:
            force = nb_forces[0]
            charges = []
            for index in range(system.getNumParticles()):
                charge, _, _ = force.getParticleParameters(index)
                charges.append(charge.value_in_unit(_unit.elementary_charge))

        return charges

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
        topology_pdb: Optional[str] = None,
        write_pdb: bool = True,
        write_system_xml: bool = True,
        partial_charges_method: str = "am1bcc",
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
        topology_pdb : str, optional, default=None
            The path to the PDB file containing the topology.
            If not None, the topology is created from this file, which is assumed to contain all the molecules.
        write_pdb : bool, optional, default=True
            Save coordinates and topology to a PDB file.
        write_system_xml : bool, optional, default=False
            Save the OpenMM system to an XML file.
        partial_charges_method : str, optional, default="am1bcc"
            The method to use for assigning partial charges to the ligand.
            See: https://docs.openforcefield.org/projects/toolkit/en/latest/api/generated/openff.toolkit.topology.Molecule.html#openff.toolkit.topology.Molecule.assign_partial_charges
        ligand_geometry : str, optional, default=None
            The geometry of the ligand.
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
            mdconfig_dict = _deepcopy(self._MDCONFIG_DICT) if mdconfig_dict is None else _deepcopy(mdconfig_dict)
        else:
            mdconfig_dict = _deepcopy(self._MDCONFIG_DICT_VACUUM) if mdconfig_dict is None else _deepcopy(mdconfig_dict)

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

        # Generate conformers for the ligand and assign partial charges
        if molecules["ligand"] is not None:
            if sdf_file_ligand is None:
                # Only generate conformers if no SDF file is provided
                # Otherwise, the geometry is taken from the SDF file
                logger.debug("Generating conformers for the ligand.")
                molecules["ligand"].generate_conformers()
            else:
                logger.debug(f"Using provided ligand geometry from {sdf_file_ligand}")
            molecules["ligand"].assign_partial_charges(partial_charges_method)

        # for m in ["ligand", "solvent"]:
        #     molecules[m].generate_conformers()

        if topology_pdb:
            logger.debug("Creating topology from PDB file.")
            mols = [mol for _, mol in molecules.items() if mol is not None]
            topology_off = _Topology.from_pdb(topology_pdb, unique_molecules=mols)
        else:
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

        if isinstance(hydrogen_mass, _unit.Quantity):
            hmr = hydrogen_mass.value_in_unit(_unit.amu)
        elif isinstance(hydrogen_mass, float):
            hmr = hydrogen_mass
        else:
            raise ValueError("Hydrogen mass must be a float or a Quantity.")

        # Create the simulation from the Interchange object
        try:
            system = interchange.to_openmm_system(
                combine_nonbonded_forces=True,
                add_constrained_forces=True,
                hydrogen_mass=hmr,
            )
        except _UnsupportedExportError as e:
            logger.warning("The OpenFF Interchange object cannot apply HMR on models with virtual sites.")
            logger.warning(f"OpenFF error: {e}")

            system = interchange.to_openmm_system(
                combine_nonbonded_forces=True,
                add_constrained_forces=True,
            )
            # Apply HMR
            system = self._apply_hmr(interchange, system, hydrogen_mass)

        # Create barostat (only if system is periodic)
        if (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions():
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

        # Handle EMLEPotential modifications
        emle_instances = self._get_modification_instances(lambda_schedule, "EMLEPotential")
        if emle_instances:
            import numpy as _np
            import sire as _sr

            # For now, we apply the same kwargs to all instances
            # In the future, this could be made per-instance specific
            for modification_name in emle_instances:
                modifications_kwargs[modification_name] = modifications_kwargs.get(modification_name, {})

            # Write .top and .gro files via the OpenFF interchange
            if _os.path.exists(self._TMP_DIR):
                _shutil.rmtree(self._TMP_DIR)
            _os.makedirs(self._TMP_DIR, exist_ok=True)
            files_prefix = _os.path.join(self._TMP_DIR, "interchange")
            interchange.to_gro(files_prefix + ".gro")
            interchange.to_top(files_prefix + ".top")

            # Read back those files using Sire
            sr_mols = _sr.load(files_prefix + ".top", files_prefix + ".gro", show_warnings=True)

            # Select the alchemical subsystem
            alchemical_subsystem = sr_mols.atoms(alchemical_atoms)

            # Write the alchemical subsystem and full system parm7 to temp files
            alchemical_prm7 = _sr.save(
                alchemical_subsystem,
                directory=self._TMP_DIR,
                filename="alchemical_subsystem.prm7",
                format=["prm7"],
            )

            # Get the original charges of the OpenMM system
            openmm_charges = self._get_openmm_charges(system)

            # Add required EMLEPotential kwargs to all instances
            for modification_name in emle_instances:
                modifications_kwargs[modification_name]["mols"] = sr_mols
                modifications_kwargs[modification_name]["parm7"] = alchemical_prm7[0]
                # TODO: uncomment for EMLE+
                modifications_kwargs[modification_name]["top_file"] = files_prefix + ".top"
                modifications_kwargs[modification_name]["crd_file"] = files_prefix + ".gro"
                modifications_kwargs[modification_name]["mm_charges"] = _np.asarray(
                    [atom.charge().value() for atom in sr_mols.atoms(alchemical_atoms)]
                )
                modifications_kwargs[modification_name]["openmm_charges"] = openmm_charges

        # Handle ML-related modifications
        ml_types = ["MLPotential", "MLInterpolation", "MLCorrection"]
        ml_instances = []
        for ml_type in ml_types:
            ml_instances.extend(self._get_modification_instances(lambda_schedule, ml_type))

        if ml_instances:
            modifications_kwargs["MLPotential"] = modifications_kwargs.get("MLPotential", {})
            modifications_kwargs["MLPotential"]["topology"] = topology

        # Handle CustomLJ modifications
        customlj_instances = self._get_modification_instances(lambda_schedule, "CustomLJ")
        if customlj_instances:
            for modification_name in customlj_instances:
                modifications_kwargs[modification_name] = modifications_kwargs.get(modification_name, {})
                modifications_kwargs[modification_name]["original_offxml"] = ffs
                modifications_kwargs[modification_name]["topology_off"] = topology_off
                modifications_kwargs[modification_name]["positions"] = positions

        # Handle ChargeTransfer modifications
        chargetransfer_instances = self._get_modification_instances(lambda_schedule, "ChargeTransfer")
        if chargetransfer_instances:
            for modification_name in chargetransfer_instances:
                modifications_kwargs[modification_name] = modifications_kwargs.get(modification_name, {})
                modifications_kwargs[modification_name]["original_offxml"] = ffs
                modifications_kwargs[modification_name]["topology_off"] = topology_off

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
            assert isinstance(integrator, _mm.Integrator), "integrator must be an OpenMM Integrator."
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

        if write_system_xml:
            with open(_os.path.join(self._TMP_DIR, "system.xml"), "w") as f:
                f.write(_mm.XmlSerializer.serialize(system))

        # Clean up the temporary directory
        if not keep_tmp_files:
            _shutil.rmtree(self._TMP_DIR)

        logger.debug("Alchemical state created successfully.")
        logger.debug("=" * 100)

        return alc_state
