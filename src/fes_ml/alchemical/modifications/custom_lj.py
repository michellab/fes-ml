from .lj_softcore import LJSoftCoreModification
import openmm.unit as _unit
import openmm as _mm
from openff.toolkit.typing.engines.smirnoff import ForceField as _ForceField
from openff.toolkit.topology import Topology as _Topology

from .base_modification import BaseModification, BaseModificationFactory
import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

class CustomLJModificationFactory(BaseModificationFactory):
    """Factory for creating CustomLJModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """
        Create an instance of CustomLJModification.

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
        args : tuple
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        openmm.System
            The modified system.
        """
        forces = {force.__class__.__name__: force for force in system.getForces()}
        custom_nb_force = forces["CustomNonbondedForce"]

        # Create a dictionary with the optimized Lennard-Jones parameters for each atom type
        force_field_opt = _ForceField(lj_offxml)
        opt_params = {}
        for p in force_field_opt.get_parameter_handler("vdW"):
            opt_params[p.id] = {
                "epsilon": p.epsilon.to_openmm().value_in_unit(_unit.kilojoules_per_mole), 
                "sigma": p.sigma.to_openmm().value_in_unit(_unit.nanometer)
                }

        # Update the Lennard-Jones parameters in the CustomNonbondedForce 
        force_field = _ForceField(*original_offxml)
        labels = force_field.label_molecules(topology_off)
        index = 0
        for mol in labels:
            for _, val in mol["vdW"].items():
                atom_type = val.id
                # Get the original Lennard-Jones parameters
                parameters = [opt_params[atom_type]["sigma"], opt_params[atom_type]["epsilon"]]
                # Update the Lennard-Jones parameters
                custom_nb_force.setParticleParameters(index, parameters)
                index += 1

        return system