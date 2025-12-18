import logging
from typing import List, Optional, Union

import openmm as _mm
import openmm.unit as _unit
from openff.toolkit.topology import Topology as _Topology
from openff.toolkit.typing.engines.smirnoff import ForceField as _ForceField

from .base_modification import BaseModification, BaseModificationFactory
from .lj_softcore import LJSoftCoreModification

logger = logging.getLogger(__name__)


class CustomLJModificationFactory(BaseModificationFactory):
    """Factory for creating CustomLJModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """
        Create an instance of CustomLJModification.

        Parameters
        ----------
        modification_name : str, optional
            Custom name for this modification instance.

        Returns
        -------
        CustomLJModification
            Instance of the modification to be applied.
        """
        return CustomLJModification(*args, **kwargs)


class CustomLJModification(BaseModification):
    """Class to modify the Lennard-Jones parameters of the CustomNonbondedForce."""

    NAME = "CustomLJ"

    pre_dependencies: List[str] = [LJSoftCoreModification.NAME]
    post_dependencies: List[str] = []

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        lambda_value: Optional[Union[float, int]],
        original_offxml: List[str],
        lj_offxml: str,
        topology_off: _Topology,
        alchemical_atoms_only: bool = True,
        non_alchemical_atoms_only: bool = False,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Apply the LJ soft core modification to the system.

        Parameters
        ----------
        system : openmm.System
            The system to be modified.
        alchemical_atoms : List[int]
            A list of the indices of the alchemical atoms.
        lambda_value : float
            The value of the alchemical state parameter.
        original_offxml : List[str]
            A list of paths to the original OFFXML files.
        lj_offxml : str
            The path to the OFFXML file with the modified Lennard-Jones parameters.
        topology_off : openff.toolkit.topology.Topology
            The topology of the system.
        alchemical_atoms_only : bool
            Whether to only apply the LJ soft core modification to the alchemical atoms.
        non_alchemical_atoms_only : bool
            Whether to only apply the LJ soft core modification to the non-alchemical atoms.
        args : tuple
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        openmm.System
            The modified system.
        """
        if non_alchemical_atoms_only and alchemical_atoms_only:
            raise ValueError("Cannot set both alchemical_atoms_only and non_alchemical_atoms_only to True.")

        # Find the related LJSoftCore force based on instance naming
        alchemical_group = self.alchemical_group
        group_forces = self.find_forces_by_group(system, self.alchemical_group)
        custom_nb_forces = [force for force in group_forces if force.getName() == f"LJSoftCore:{alchemical_group}"]
        if not custom_nb_forces:
            raise ValueError(f"Attempting to modify LJ parameters but no LJSoftCore force found for alchemical group '{alchemical_group}'")
        custom_nb_force = custom_nb_forces[0]

        # Create a dictionary with the optimized Lennard-Jones parameters for each atom type
        force_field_opt = _ForceField(lj_offxml)
        opt_params = {
            p.id: {
                "epsilon": p.epsilon.to_openmm().value_in_unit(_unit.kilojoules_per_mole),
                "sigma": p.sigma.to_openmm().value_in_unit(_unit.nanometer),
            }
            for p in force_field_opt.get_parameter_handler("vdW")
        }

        # Get atom types
        labels_opt = force_field_opt.label_molecules(topology_off)

        # Get dictionary with the original Lennard-Jones parameters for each atom type
        force_field = _ForceField(*original_offxml)
        orig_params = {
            p.id: {
                "epsilon": p.epsilon.to_openmm().value_in_unit(_unit.kilojoules_per_mole),
                "sigma": p.sigma.to_openmm().value_in_unit(_unit.nanometer),
            }
            for p in force_field.get_parameter_handler("vdW")
        }

        # Get atom types
        labels_orig = force_field.label_molecules(topology_off)

        # Flatten the vdW parameters for all molecules
        opt_vdw_parameters = [
            (opt_params[val.id]["sigma"], opt_params[val.id]["epsilon"]) for mol in labels_opt for _, val in mol["vdW"].items()
        ]

        orig_vdw_parameters = [
            (orig_params[val.id]["sigma"], orig_params[val.id]["epsilon"]) for mol in labels_orig for _, val in mol["vdW"].items()
        ]

        # Combine original and optimized parameters based on lambda_value
        vdw_parameters = [
            (
                (1 - lambda_value) * orig_vdw_parameters[i][0] + lambda_value * opt_vdw_parameters[i][0],
                (1 - lambda_value) * orig_vdw_parameters[i][1] + lambda_value * opt_vdw_parameters[i][1],
            )
            for i in range(len(orig_vdw_parameters))
        ]

        # Update the Lennard-Jones parameters in a single loop
        for index, parameters in enumerate(vdw_parameters):
            if alchemical_atoms_only and index not in alchemical_atoms or non_alchemical_atoms_only and index in alchemical_atoms:
                continue
            custom_nb_force.setParticleParameters(index, parameters)

        # Update name (will override LJSoftCore name)
        custom_nb_force.setName(self.modification_name)

        return system
