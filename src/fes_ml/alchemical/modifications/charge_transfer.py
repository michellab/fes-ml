import logging
from typing import List, Optional, Union

import openmm as _mm
import openmm.unit as _unit
import openmm.app as _app
from openff.toolkit.topology import Topology as _Topology
from openff.toolkit.typing.engines.smirnoff import ForceField as _ForceField

from .base_modification import BaseModification, BaseModificationFactory

logger = logging.getLogger(__name__)


class ChargeTransferModificationFactory(BaseModificationFactory):
    """Factory for creating ChargeTransferModication instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """
        Create an instance of ChargeTransferModication.

        Returns
        -------
        ChargeTransferModication
            Instance of the modification to be applied.
        """
        return ChargeTransferModication(*args, **kwargs)


class ChargeTransferModication(BaseModification):
    """Class to create a charge transfer potential based on a CustomNonbondedForce."""

    NAME = "ChargeTransfer"

    pre_dependencies: List[str] = []
    post_dependencies: List[str] = []

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
        #
        ct_paraams = {
            "n21": {"sigma": 2.8176, "eps": 77.6383},
            "n16": {"sigma": 3.7264, "eps": 16.7750},
            "n3": {"sigma": 6.9854, "eps": 7.8837},
            "n-tip3p-O": {"sigma": 4.4462, "eps": 33.1282},
            "n-tip3p-H": {"sigma": 5.7474, "eps": 8.1208},
            "n20": {"sigma": 3.6300, "eps": 46.6064},
            "n2": {"sigma": 6.8839, "eps": 6.7772},
            "n18": {"sigma": 4.0536, "eps": 28.4463},
            "n14": {"sigma": 3.3925, "eps": 19.4895},
            "n17": {"sigma": 4.1549, "eps": 29.7470},
            "n9": {"sigma": 6.8698, "eps": 8.8237},
            "n4": {"sigma": 7.0525, "eps": 8.1752},
            "n13": {"sigma": 6.5274, "eps": 8.0680},
            "n11": {"sigma": 6.1873, "eps": 4.1690},
            "n19": {"sigma": 4.2323, "eps": 30.6547},
            "n12": {"sigma": 4.6499, "eps": 1.9570},
            "n7": {"sigma": 7.1008, "eps": 7.7616},
            "n15": {"sigma": 3.6511, "eps": 26.1448},
            "n10": {"sigma": 6.5920, "eps": 6.4696},
            "n8": {"sigma": 7.1305, "eps": 9.1567},
        }

        energy_function = f"-{lambda_value}*epsilon*exp(-sigma*r);"
        energy_function += "sigma = 0.5*(sigma1+sigma2);"
        energy_function += "epsilon = sqrt(epsilon1*epsilon2);"

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

        # Update the Lennard-Jones parameters in the CustomNonbondedForce
        force_field = _ForceField(*original_offxml)
        labels = force_field.label_molecules(topology_off)

        # Get atom types
        atom_types = [val.id for mol in labels for _, val in mol["vdW"].items()]

        for index in range(system.getNumParticles()):
            charge_transfer_force.addParticle(
                [
                    ct_paraams.get(atom_types[index], {}).get("sigma", 0) * 10,
                    ct_paraams.get(atom_types[index], {}).get("eps", 0) * 1e3,
                ]
            )

        # Set the custom force to occur between just the alchemical particle and the other particles
        mm_atoms = set(range(system.getNumParticles())) - set(alchemical_atoms)
        charge_transfer_force.addInteractionGroup(alchemical_atoms, mm_atoms)

        # Add the CustomNonbondedForce to the System
        system.addForce(charge_transfer_force)

        return system
