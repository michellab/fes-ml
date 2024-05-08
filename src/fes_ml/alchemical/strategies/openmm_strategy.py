"""Sire alchemical state creation strategy."""

# OpenMM imports
import logging
import shutil as _shutil

# other imports
from copy import deepcopy
from copy import deepcopy as _deepcopy
from typing import Any, Dict, Iterable, List, Optional, Union

import openmm as _mm
import openmm.app as _app
import openmm.unit as unit

# mace imports
import openmmml.models as models
from openff.interchange import Interchange
from openff.interchange.components._packmol import UNIT_CUBE, pack_box

# OpenFF-toolkit imports
from openff.toolkit import ForceField, Molecule
from openff.toolkit import Topology as offTopology
from openff.units import unit as offunit
from openff.units.openmm import to_openmm as offquantity_to_openmm

# OpenMM ForceFields imports
from openmmforcefields.generators import (
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
)

# OpenMM-Torch imports
from openmmtorch import TorchForce

from ...utils import energy_decomposition as energy_decomposition
from ..alchemical_state import AlchemicalState
from .alchemical_functions import alchemify as alchemify
from .base_strategy import AlchemicalStateCreationStrategy

logger = logging.getLogger(__name__)


class OpenMMCreationStrategy(AlchemicalStateCreationStrategy):
    """Strategy for creating alchemical states using OpenMM."""

    @staticmethod
    def removeBonds(
        system: _mm.System,
        atoms: Iterable[int],
        removeInSet: bool,
        removeConstraints: bool,
    ) -> _mm.System:
        """Copy a System, removing all bonded interactions between atoms in (or not in) a particular set.

        Parameters
        ----------
        system: System
            the System to copy
        atoms: Iterable[int]
            a set of atom indices
        removeInSet: bool
            if True, any bonded term connecting atoms in the specified set is removed.  If False,
            any term that does *not* connect atoms in the specified set is removed
        removeConstraints: bool
            if True, remove constraints between pairs of atoms in the set

        Returns
        -------
        a newly created System object in which the specified bonded interactions have been removed
        """
        atomSet = set(atoms)

        print(atomSet)

        # Create an XML representation of the System.

        import xml.etree.ElementTree as ET

        xml = _mm.XmlSerializer.serialize(system)
        root = ET.fromstring(xml)

        # This function decides whether a bonded interaction should be removed.

        def shouldRemove(termAtoms):
            return all(a in atomSet for a in termAtoms) == removeInSet

        # Remove bonds, angles, and torsions.

        for bonds in root.findall("./Forces/Force/Bonds"):
            for bond in bonds.findall("Bond"):
                bondAtoms = [int(bond.attrib[p]) for p in ("p1", "p2")]
                if shouldRemove(bondAtoms):
                    bonds.remove(bond)
        for angles in root.findall("./Forces/Force/Angles"):
            for angle in angles.findall("Angle"):
                angleAtoms = [int(angle.attrib[p]) for p in ("p1", "p2", "p3")]
                if shouldRemove(angleAtoms):
                    angles.remove(angle)
        for torsions in root.findall("./Forces/Force/Torsions"):
            for torsion in torsions.findall("Torsion"):
                torsionAtoms = [
                    int(torsion.attrib[p]) for p in ("p1", "p2", "p3", "p4")
                ]
                if shouldRemove(torsionAtoms):
                    torsions.remove(torsion)

        # Optionally remove constraints.

        if removeConstraints:
            for constraints in root.findall("./Constraints"):
                for constraint in constraints.findall("Constraint"):
                    constraintAtoms = [int(constraint.attrib[p]) for p in ("p1", "p2")]
                    if shouldRemove(constraintAtoms):
                        constraints.remove(constraint)

        # Create a new System from it.

        return _mm.XmlSerializer.deserialize(ET.tostring(root, encoding="unicode"))

    def create_solvated(
        self, water_model=None, padding=2.0, ionicStrength=0.15, **kwargs
    ):
        print(f"solvating...")

        # make an OpenFF Topology of the ligand
        ligand_off_topology = offTopology.from_molecules(molecules=[self.ligand])

        # convert it to an OpenMM Topology
        ligand_omm_topology = ligand_off_topology.to_openmm()

        # get the positions of the ligand
        ligand_positions = offquantity_to_openmm(self.ligand.conformers[0])

        if water_model.upper() == "TIP3P":
            # Create an OpenMM ForceField object for water
            ff = _app.ForceField("amber14/tip3p.xml")

            # add in the SMIRNOFF template generator
            ff.registerTemplateGenerator(self.smirnoff.generator)

            # create an OpenMM Modeller object
            modeller = _app.Modeller(ligand_omm_topology, ligand_positions)
            # solvate
            modeller.addSolvent(
                ff,
                padding=padding * unit.nanometer,
                ionicStrength=ionicStrength * unit.molar,
            )

        # TODO some way to calc no of water molecules based on padding so consistent.
        # TODO possible to also calc ionic strength below with OPC?
        elif water_model.upper() == "OPC":
            water = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
            water.generate_conformers()

            ff = ForceField(self.ligand_forcefield, "opc.offxml")
            topology = pack_box(
                molecules=[self.ligand, water],
                number_of_copies=[1, 900],
                mass_density=1.0 * offunit.gram / offunit.milliliter,
                box_shape=UNIT_CUBE,
            )

            interchange = Interchange.from_smirnoff(force_field=ff, topology=topology)
            modeller = _app.Modeller(
                topology.to_openmm(), interchange.positions.to_openmm()
            )
            # self.system = interchange.to_openmm()
            ff = _app.ForceField("amber14/opc.xml")
            ff.registerTemplateGenerator(self.smirnoff.generator)
            # need to add extra particles for which element is None (virtual sites)
            modeller.addExtraParticles(ff)

        else:
            logging.error("please state the water model.")
            return

        self.modeller = modeller
        self.ff = ff
        self.leg = "solvated"

    def create_vacuum(self):
        # Create an OpenMM ForceField object
        ff = _app.ForceField()

        # add in the SMIRNOFF template generator
        ff.registerTemplateGenerator(self.smirnoff.generator)

        # make an OpenFF Topology of the ligand
        ligand_off_topology = offTopology.from_molecules(molecules=[self.ligand])

        # convert it to an OpenMM Topology
        ligand_omm_topology = ligand_off_topology.to_openmm()

        # get the positions of the ligand
        ligand_positions = offquantity_to_openmm(self.ligand.conformers[0])

        # create an OpenMM Modeller object
        modeller = _app.Modeller(ligand_omm_topology, ligand_positions)

        self.modeller = modeller
        self.ff = ff
        self.leg = "vacuum"

    def create_system(
        self, ML=True, ml_atoms=None, model="ani2x", leg="solvated", **kwargs
    ):
        if leg.lower() == "solvated":
            self.create_solvated(**kwargs)
        elif leg.lower() == "vacuum":
            self.create_vacuum()
        else:
            raise ValueError(f"{leg} must be 'solvated' or 'bound'.")

        if "nonbondedMethod" not in kwargs.keys():
            if self.leg == "vacuum":
                kwargs["nonbondedMethod"] = _app.NoCutoff
            elif self.leg == "solvated":
                kwargs["nonbondedMethod"] = _app.PME
            else:
                print(
                    "please create the system using create_vacuum or create_solvated. Not creating the system..."
                )
                return

        # create system with all MM forces
        import inspect

        key_list = []
        for key in kwargs.keys():
            if key in inspect.signature(_app.ForceField.createSystem).parameters:
                key_list.append(key)
        system_kwarg_dict = {key: kwargs[key] for key in key_list}

        self.system = self.ff.createSystem(self.modeller.topology, **system_kwarg_dict)

        if ml_atoms:
            self.ml_atoms = ml_atoms
        else:
            self.ml_atoms = [
                atom.index for atom in list(self.modeller.topology.chains())[0].atoms()
            ]
            if len(self.ml_atoms) > 100:
                raise ValueError(
                    f"The length of the found ml region is >100. This seems like too much."
                )

        if self.leg == "vacuum":
            pass
        elif self.leg == "solvated":
            self.system.addForce(
                _mm.MonteCarloBarostat(1.0 * unit.bar, float(self.temperature))
            )
        else:
            print(
                "please create the system using create_vacuum or create_solvated. Not creating the system..."
            )
            return

        if ML == True:
            self.create_ml(model)

    def _create_ani2x(self, system):
        ml_system = deepcopy(system)

        macepotiml = models.anipotential.ANIPotentialImpl(name="ani2x")
        macepotiml.addForces(
            self.modeller.topology,
            ml_system,
            self.ml_atoms,
            0,
            returnEnergyType="energy",
            periodic=True,
        )

        torch_force = [
            TorchForce.cast(f)
            for f in ml_system.getForces()
            if TorchForce.isinstance(f)
        ][0]

        self.ml_force = deepcopy(torch_force)

    def _create_mace(self, system):
        ml_system = deepcopy(system)

        macepotiml = models.macepotential.MACEPotentialImpl(
            name="mace-off23-small", modelPath=""
        )
        macepotiml.addForces(
            self.modeller.topology,
            ml_system,
            self.ml_atoms,
            0,
            returnEnergyType="energy",
            periodic=True,
        )

        torch_force = [
            TorchForce.cast(f)
            for f in ml_system.getForces()
            if TorchForce.isinstance(f)
        ][0]

        self.ml_force = deepcopy(torch_force)

    def create_ml(self, model="ani2x"):
        if model == "ani2x":
            self._create_ani2x(self.system)
        elif model == "mace":
            self._create_mace(self.system)

        self._create_custom_force(self.system)

    def _create_custom_force(self, system):
        # create the MM forces for just the ligand

        # make a customCV force so we can take away the ligand bonded forces from the system
        cv = _mm.CustomCVForce("")

        # create a new system that contains just the bonded MM forces of the ligand
        ligand_bonded_system = OpenMMCreationStrategy.removeBonds(
            system, self.ml_atoms, False, False
        )

        # extract the ligand MM bonded forces
        ligand_bonded_forces = []
        for force in ligand_bonded_system.getForces():
            if (
                hasattr(force, "addBond")
                or hasattr(force, "addAngle")
                or hasattr(force, "addTorsion")
            ):
                ligand_bonded_forces.append(force)

        # add them to the CV force
        mm_ligand_force_names = []
        for i, force in enumerate(ligand_bonded_forces):
            name = f"mmForce{i+1}"
            cv.addCollectiveVariable(name, deepcopy(force))
            mm_ligand_force_names.append(name)

        # setup the intra ligand MM non-bonded forces
        # use a custom bond force for the coulomb + LJ interaction.
        for force in system.getForces():
            if isinstance(force, _mm.NonbondedForce):
                internalNonbonded = _mm.CustomBondForce(
                    "138.935456*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)"
                )
                internalNonbonded.addPerBondParameter("chargeProd")
                internalNonbonded.addPerBondParameter("sigma")
                internalNonbonded.addPerBondParameter("epsilon")
                numParticles = system.getNumParticles()
                atomCharge = [0] * numParticles
                atomSigma = [0] * numParticles
                atomEpsilon = [0] * numParticles
                for i in range(numParticles):
                    charge, sigma, epsilon = force.getParticleParameters(i)
                    atomCharge[i] = charge
                    atomSigma[i] = sigma
                    atomEpsilon[i] = epsilon
                exceptions = {}
                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                    exceptions[(p1, p2)] = (chargeProd, sigma, epsilon)
                for p1 in self.ml_atoms:
                    for p2 in self.ml_atoms:
                        if p1 == p2:
                            break
                        if (p1, p2) in exceptions:
                            chargeProd, sigma, epsilon = exceptions[(p1, p2)]
                        elif (p2, p1) in exceptions:
                            chargeProd, sigma, epsilon = exceptions[(p2, p1)]
                        else:
                            chargeProd = atomCharge[p1] * atomCharge[p2]
                            sigma = 0.5 * (atomSigma[p1] + atomSigma[p2])
                            epsilon = unit.sqrt(atomEpsilon[p1] * atomEpsilon[p2])
                        if chargeProd._value != 0 or epsilon._value != 0:
                            internalNonbonded.addBond(
                                p1, p2, [chargeProd, sigma, epsilon]
                            )

                # add the non bonded forces to the cv
                if internalNonbonded.getNumBonds() > 0:
                    name = f"mmForce{len(mm_ligand_force_names)+1}"
                    cv.addCollectiveVariable(name, internalNonbonded)
                    mm_ligand_force_names.append(name)

        # set the CV force to be the correction term V_ml(ligand) - Vmm(ligand)
        cv.addCollectiveVariable("ml_force", self.ml_force)

        cv_force_string = "ml_force - 1.0*(" + "+".join(mm_ligand_force_names) + ")"

        print(cv_force_string)
        cv.setEnergyFunction(cv_force_string)
        system.addForce(cv)
        # the cv includes the mmForces (ligand bonded, internal nonbonded), the ml_forces, and an energy function that incl the ml_forces-(mm forces)

    def create_alchemical_state(
        self,
        sdf_file: str,
        alchemical_atoms: List[int],
        lambda_lj: Union[float, None],
        lambda_q: Union[float, None],
        lambda_interpolate: Union[float, None],
        lambda_ml_correction: Union[float, None],
        lambda_emle: Union[float, None],
        minimise_iterations: int = 1,
        ml_potential: str = "ani2x",
        ml_potential_kwargs: Optional[Dict[str, Any]] = None,
        create_system_kwargs: Optional[Dict[str, Any]] = None,
        dynamics_kwargs: Optional[Dict[str, Any]] = None,
        emle_kwargs: Optional[Dict[str, Any]] = None,
        integrator: Optional[Any] = None,
        **args,
    ) -> AlchemicalState:
        """
        Create an alchemical state for the given lambda values using OpenMM.

        Parameters
        ----------
        sdf_file : str
            Path to the sdf file.
        alchemical_atoms : list of int
            A list of atom indices to be alchemically modified.
        lambda_lj : float or None
            The lambda value for the softcore Lennard-Jones potential.
        lambda_q : float or None
            The lambda value to scale the charges.
        lambda_interpolate : float or None
            The lambda value to interpolate between the ML and MM potentials in a mechanical embedding scheme.
        lambda_emle : float or None
            The lambda value to interpolate between the ML and MM potentials in a electrostatic embedding scheme.
        minimise_iterations : int, optional, default=1
            The number of minimisation iterations to perform before creating the alchemical state.
            1 step is enough to bring the geometry to the distances imposed by the restraints.
            If None, no minimisation is performed.
        ml_potential : str, optional, default='ani2x'
            The machine learning potential to use in the mechanical embedding scheme.
        ml_potential_kwargs : dict, optional, default=None
            Additional keyword arguments to be passed to MLPotential when creating the ML potential in OpenMM-ML.
            See: https://openmm.github.io/openmm-ml/dev/generated/openmmml.MLPotential.html
        create_system_kwargs : dict, optional, default=None
            Additional keyword arguments to be passed when creating the system.
        dynamics_kwargs : dict
            Additional keyword arguments to be used.
        integrator : Any, optional, default=None
            The OpenMM integrator to use. If None, the integrator is the one used in the dynamics_kwargs, if provided.
            Otherwise, the default is a LangevinMiddle integrator with a 1 fs timestep and a 298.15 K temperature.

        Returns
        -------
        AlchemicalState
            The alchemical state.
        """

        if lambda_interpolate is not None and lambda_emle is not None:
            raise ValueError(
                "The lambda_interpolate and lambda_emle parameters are mutually exclusive."
            )

        if any([lambda_interpolate, lambda_emle]) and lambda_ml_correction is not None:
            raise ValueError(
                "The lambda_ml_correction parameter is not compatible with lambda_interpolate or lambda_emle."
            )

        create_system_kwargs_default = {
            "ligand_forcefield": "openff",
            "water_model": "OPC",
            "HMR": False,
        }
        if create_system_kwargs is None:
            create_system_kwargs = create_system_kwargs_default
        else:
            # add any missing kwargs
            for key in create_system_kwargs_default.keys():
                if key not in create_system_kwargs:
                    create_system_kwargs[key] = create_system_kwargs_default[key]
            create_system_kwargs = _deepcopy(create_system_kwargs)

        if dynamics_kwargs is None:
            dynamics_kwargs = {
                "timestep": "1fs",
                "constraint": "none",
                "integrator": "langevin_middle",
                "temperature": "298.15K",  # Kelvin
                "HMR": False,
            }
        else:
            dynamics_kwargs = _deepcopy(dynamics_kwargs)

        self.temperature = dynamics_kwargs["temperature"][:-1]

        logger.debug("-" * 100)
        logger.debug("Creating alchemical state using OpenMMCreationStrategy.")
        logger.debug(f"sdf_file: {sdf_file}")
        logger.debug(f"alchemical_atoms: {alchemical_atoms}")
        logger.debug(f"lambda_lj: {lambda_lj}")
        logger.debug(f"lambda_q: {lambda_q}")
        logger.debug(f"lambda_interpolate: {lambda_interpolate}")
        # TODO this needs to be one ???
        logger.debug(f"lambda_ml_correction: {lambda_ml_correction}")
        logger.debug(f"ml_potential: {ml_potential}")
        logger.debug("dynamics_kwargs:")
        for key, value in dynamics_kwargs.items():
            logger.debug(f"{key}: {value}")
        for key, value in create_system_kwargs.items():
            logger.debug(f"{key}: {value}")

        # Load the ligand
        self.ligand = Molecule.from_file(sdf_file)

        # Create the SMIRNOFF template generator with the default installed force field
        if create_system_kwargs["ligand_forcefield"].upper() == "OPENFF":
            self.smirnoff = SMIRNOFFTemplateGenerator(
                molecules=self.ligand,
                # forcefield="", # 'openff-2.1.0' "openff_unconstrained-2.0.0.offxml"
            )
            self.ligand_forcefield = self.smirnoff.smirnoff_filename
        elif create_system_kwargs["ligand_forcefield"].upper() == "GAFF":
            self.smirnoff = GAFFTemplateGenerator(
                molecules=self.ligand,
                # forcefield='gaff-2.11'
            )
            self.ligand_forcefield = self.smirnoff.gaff_version

        if "hydrogenMass" not in create_system_kwargs.keys():
            if create_system_kwargs["HMR"]:
                create_system_kwargs["hydrogenMass"] = 3 * unit.amu
            else:
                pass

        # if "constraints" not in dynamics_kwargs.keys():
        #     dynamics_kwargs["constraints"] = self.constrain_H

        # if constrain_H:
        #     self.constrain_H = "HBonds"
        # else:
        #     self.constrain_H = None

        self.modeller = None
        self.ff = None
        self.leg = None
        self.system = None

        self.create_system(
            ML=True,
            model=ml_potential,
            ml_atoms=alchemical_atoms,
            **create_system_kwargs,
        )

        # alchemify

        if integrator is None:
            raise ValueError("please provide an integrator")
        else:
            integrator = _deepcopy(integrator)

        # Create a new context and set positions and velocities
        simulation = _app.Simulation(
            self.modeller.topology,
            self.system,
            integrator,  # omm.getPlatform() # TODO this was from the emle part?
        )
        simulation.context.setPositions(self.modeller.positions)
        try:
            simulation.context.setVelocitiesToTemperature(integrator.getTemperature())
        except AttributeError:
            simulation.context.setVelocitiesToTemperature(
                float(dynamics_kwargs["temperature"][:-1])
            )

        logger.debug("Energy decomposition of the system:")
        logger.debug(
            f"Total potential energy: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}"
        )
        energy_decomp = energy_decomposition(self.system, simulation.context)
        for force, energy in energy_decomp.items():
            logger.debug(f"{force}: {energy}")
        logger.debug("Alchemical state created successfully.")
        logger.debug("-" * 100)

        # Create the AlchemicalState
        alc_state = AlchemicalState(
            system=self.system,
            context=simulation.context,
            integrator=integrator,
            simulation=simulation,
            topology=self.modeller.topology,
            lambda_lj=lambda_lj,
            lambda_q=lambda_q,
            lambda_interpolate=lambda_interpolate,
            lambda_emle=lambda_emle,
            lambda_ml_correction=lambda_ml_correction,
        )

        return alc_state
