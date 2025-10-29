import logging
from typing import List, Optional, Union

import openmm as _mm
import openmm.unit as _unit
from openff.toolkit.topology import Topology as _Topology
from openff.toolkit.typing.engines.smirnoff import ForceField as _ForceField

from .base_modification import BaseModification, BaseModificationFactory

logger = logging.getLogger(__name__)


class ChargeTransferModificationFactory(BaseModificationFactory):
    """Factory for creating ChargeTransferModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """
        Create an instance of ChargeTransferModification.

        Returns
        -------
        ChargeTransferModification
            Instance of the modification to be applied.
        """
        return ChargeTransferModification(*args, **kwargs)


class ChargeTransferModification(BaseModification):
    """Class to create a charge transfer potential based on a CustomNonbondedForce."""

    NAME = "ChargeTransfer"

    pre_dependencies: List[str] = []
    post_dependencies: List[str] = []

    @staticmethod
    def get_is_donor_acceptor(
        topology: _Topology, alchemical_atoms: List[int], symmetric: bool = False
    ) -> tuple[list[int], list[int]]:
        """
        Generate per-atom flags for donors and acceptors.

        Parameters
        ----------
        topology : openff.toolkit.topology.Topology

        alchemical_atoms : list of int
            List of indices of alchemical atoms.

        symmetric : bool, optional
            If True, CT will be applied symmetrically between alchemical and MM atoms.
            Otherwise, only from MM to alchemical atoms. Default is False.

        Returns
        -------
        is_donor : list[int]
            1 if the atom is a hydrogen bonded to N/O, 0 otherwise.
        is_acceptor : list[int]
            1 if the atom is N or O, 0 otherwise.
        """
        n_atoms = topology.n_atoms
        is_donor = [0] * n_atoms
        is_acceptor = [0] * n_atoms

        for idx, atom in enumerate(topology.atoms):
            # Donor: hydrogen bonded to N/O/S
            if atom.atomic_number == 1 and (symmetric or idx in alchemical_atoms):
                for bonded_atom in atom.bonded_atoms:
                    if bonded_atom.atomic_number in [7, 8, 16]:
                        is_donor[idx] = 1
                        break
            # Acceptor: heavy atoms
            elif atom.atomic_number == 8 and (symmetric or idx not in alchemical_atoms):
                #num_H = sum(b.atomic_number == 1 for b in atom.bonded_atoms)
                is_acceptor[idx] = 1
            elif atom.atomic_number == 7:
                # Nitrogen
                #num_H = sum(b.atomic_number == 1 for b in atom.bonded_atoms)
                is_acceptor[idx] = 1
                # if num_H != 1 and num_H != 2:  # optionally exclude primary/secondary amines
                #    is_acceptor[idx] = 1
                # elif num_H == 0:  # tertiary amine
                #    is_acceptor[idx] = 1

        return is_donor, is_acceptor

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        original_offxml: List[str],
        ct_offxml: str,
        topology_off: _Topology,
        lambda_value: Optional[Union[float, int]] = 1.0,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Apply the LJ soft core modification to the system.

        Parameters
        ----------
        system : openmm.System
            The system to be modified.
        alchemical_atoms : list of int
            The indices of the alchemical atoms in the system.
        ct_offxml: str
            Path to the offxml file containing the charge transfer parameters.
        original_offxml : List[str]
            List of paths to the original offxml files.
        topology_off : openff.toolkit.topology.Topology
            The OpenFF Topology object of the system.
        lambda_value : float
            The value of the alchemical state parameter.
        args : tuple
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        openmm.System
            The modified system.
        """
        # Convert units
        energy_function = f"-{lambda_value}*donor_acceptor*epsilon*exp(-r/sigma);"
        energy_function += "sigma = sqrt(sigma1*sigma2);"
        energy_function += "epsilon = (epsilon1*epsilon2);"
        energy_function += (
            "donor_acceptor = 1;"#isDonor1*isAcceptor2 + isDonor2*isAcceptor1;"
        )

        logger.debug(f"Charge transfer function: {energy_function}")

        # Create a CustomNonbondedForce to compute the CT
        charge_transfer_force = _mm.CustomNonbondedForce(energy_function)
        charge_transfer_force.setNonbondedMethod(2)  # CutoffPeriodic
        charge_transfer_force.setCutoffDistance(0.6 * _unit.nanometer)
        charge_transfer_force.setUseSwitchingFunction(True)
        charge_transfer_force.setSwitchingDistance(0.5 * _unit.nanometer)
        charge_transfer_force.setUseLongRangeCorrection(False)

        # Add per-particle parameters to the CustomNonbondedForce
        charge_transfer_force.addPerParticleParameter("sigma")
        charge_transfer_force.addPerParticleParameter("epsilon")
        #charge_transfer_force.addPerParticleParameter("isDonor")
        #charge_transfer_force.addPerParticleParameter("isAcceptor")

        # Update the Lennard-Jones parameters in the CustomNonbondedForce
        #force_field = _ForceField(*original_offxml)
        #labels = force_field.label_molecules(topology_off)

        # Get atom types
        #atom_types = [val.id for mol in labels for _, val in mol["vdW"].items()]

        # Get donor/acceptor flags
        is_donor, is_acceptor = ChargeTransferModification.get_is_donor_acceptor(
            topology_off, alchemical_atoms
        )

        # CT force field
        ct_force_field = _ForceField(ct_offxml)
        labels = ct_force_field.label_molecules(topology_off)
        ct_params = {
            p.id: {
                "epsilon": p.epsilon.to_openmm().value_in_unit(
                    _unit.kilojoules_per_mole
                ),
                "sigma": p.sigma.to_openmm().value_in_unit(_unit.nanometer),
            }
            for p in ct_force_field.get_parameter_handler("vdW")
        }
        atom_types = [val.id for mol in labels for _, val in mol["vdW"].items()]

        for index in range(system.getNumParticles()):
            at_type = atom_types[index]
            charge_transfer_force.addParticle(
                [
                    ct_params[at_type]["sigma"],
                    ct_params[at_type]["epsilon"],# * 10.0,
                    #is_donor[index],
                    #is_acceptor[index],
                ]
            )

        # Set the custom force to occur between just the alchemical particle and the other particles
        mm_atoms = set(range(system.getNumParticles())) - set(alchemical_atoms)
        charge_transfer_force.addInteractionGroup(alchemical_atoms, mm_atoms)

        # Add the CustomNonbondedForce to the System
        system.addForce(charge_transfer_force)

        return system
