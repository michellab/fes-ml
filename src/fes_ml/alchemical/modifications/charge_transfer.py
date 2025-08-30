import logging
from typing import List, Optional, Union

import openmm as _mm
import openmm.app as _app
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
    def get_is_donor_acceptor(topology: _Topology) -> tuple[list[int], list[int]]:
        """
        Generate per-atom flags for donors and acceptors.

        Parameters
        ----------
        topology : openff.toolkit.topology.Topology

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
            if atom.atomic_number == 1:
                for bonded_atom in atom.bonded_atoms:
                    if bonded_atom.atomic_number in [7, 8, 16]:
                        is_donor[idx] = 1
                        break
            # Acceptor: heavy atoms
            elif atom.atomic_number == 8:
                num_H = sum(b.atomic_number == 1 for b in atom.bonded_atoms)
                is_acceptor[idx] = 1
            elif atom.atomic_number == 7:
                # Nitrogen
                num_H = sum(b.atomic_number == 1 for b in atom.bonded_atoms)
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
        ct_paraams = {
            "n21": {"sigma": 3.1739, "eps": 169.0640},
            "n16": {"sigma": 4.3983, "eps": 42.4290},
            "n3": {"sigma": 5.9554, "eps": 0.1890},
            "n-tip3p-O": {"sigma": 5.5019, "eps": 119.9913},
            "n-tip3p-H": {"sigma": 5.6480, "eps": 3.4209},
            "n20": {"sigma": 4.1808, "eps": 71.9843},
            "n2": {"sigma": 7.6786, "eps": 0.1502},
            "n18": {"sigma": 4.7983, "eps": 78.6935},
            "n14": {"sigma": 3.9557, "eps": 50.0886},
            "n17": {"sigma": 4.6716, "eps": 61.3267},
            "n9": {"sigma": 5.2866, "eps": 0.8104},
            "n4": {"sigma": 5.8806, "eps": 0.4651},
            "n13": {"sigma": 5.3179, "eps": 1.1033},
            "n11": {"sigma": 5.2541, "eps": 0.7611},
            "n19": {"sigma": 4.9034, "eps": 75.1253},
            "n12": {"sigma": 4.6507, "eps": 1.0854},
            "n7": {"sigma": 6.8461, "eps": 0.5855},
            "n15": {"sigma": 4.3835, "eps": 59.0651},
            "n10": {"sigma": 5.3841, "eps": 0.4192},
            "n8": {"sigma": 6.7245, "eps": 0.7046},
        }

        energy_function = f"-{lambda_value}*donor_acceptor*epsilon*exp(-sigma*r);"
        energy_function += "sigma = 0.5*(sigma1+sigma2);"
        energy_function += "epsilon = sqrt(epsilon1*epsilon2);"
        energy_function += "donor_acceptor = isDonor1*isAcceptor2;"

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
        charge_transfer_force.addPerParticleParameter("isDonor")
        charge_transfer_force.addPerParticleParameter("isAcceptor")

        # Update the Lennard-Jones parameters in the CustomNonbondedForce
        force_field = _ForceField(*original_offxml)
        labels = force_field.label_molecules(topology_off)

        # Get atom types
        atom_types = [val.id for mol in labels for _, val in mol["vdW"].items()]

        # Get donor/acceptor flags
        is_donor, is_acceptor = ChargeTransferModification.get_is_donor_acceptor(
            topology_off
        )

        for index in range(system.getNumParticles()):
            charge_transfer_force.addParticle(
                [
                    ct_paraams.get(atom_types[index], {}).get("sigma", 0) * 10,
                    ct_paraams.get(atom_types[index], {}).get("eps", 0) * 1e3,
                    is_donor[index],
                    is_acceptor[index],
                ]
            )

        # Set the custom force to occur between just the alchemical particle and the other particles
        mm_atoms = set(range(system.getNumParticles())) - set(alchemical_atoms)
        charge_transfer_force.addInteractionGroup(alchemical_atoms, mm_atoms)

        # Add the CustomNonbondedForce to the System
        system.addForce(charge_transfer_force)

        return system
