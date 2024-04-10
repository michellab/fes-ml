"""This module contains the implementation of the IntraMolecularNonBondedModification class."""

import logging
from typing import List

import openmm as _mm
import openmm.unit as _unit

from ..alchemist import Alchemist
from .base_modification import BaseModification, BaseModificationFactory

logger = logging.getLogger(__name__)


class IntraMolecularNonBondedExceptionsModificationFactory(BaseModificationFactory):
    """Factory for creating IntraMolecularNonBondedModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of IntraMolecularNonBondedModification.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        IntraMolecularNonBondedExceptionsModification
            The modification to be applied.
        """
        return IntraMolecularNonBondedExceptionsModification(*args, **kwargs)


class IntraMolecularNonBondedForcesModificationFactory(BaseModificationFactory):
    """Factory for creating IntraMolecularNonBondedForcesModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of IntraMolecularNonBondedForcesModification.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        IntraMolecularNonBondedForcesModification
            The modification to be applied.
        """
        return IntraMolecularNonBondedForcesModification(*args, **kwargs)


class IntraMolecularNonBondedExceptionsModification(BaseModification):
    NAME = "_lambda_intramolecular_nonbonded_exceptions"

    def apply(self, system: _mm.System, alchemical_atoms: List[int]) -> _mm.System:
        """
        Add exceptions to the NonbondedForce and CustomNonbondedForces
        to prevent the alchemical atoms from interacting as these interactions
        are already taken into account by the CustomBondForce.

        Parameters
        ----------
        system : openmm.System
            The System to modify.
        alchemical_atoms : list of int
            The indices of the alchemical atoms.

        Returns
        -------
        system : openmm.System
            The modified System.
        """
        atom_list = list(alchemical_atoms)
        for force in system.getForces():
            if isinstance(force, _mm.NonbondedForce):
                for i in range(len(atom_list)):
                    for j in range(i):
                        force.addException(atom_list[i], atom_list[j], 0, 1, 0, True)
            elif isinstance(force, _mm.CustomNonbondedForce):
                existing = set(
                    tuple(force.getExclusionParticles(i))
                    for i in range(force.getNumExclusions())
                )
                for i in range(len(atom_list)):
                    a1 = atom_list[i]
                    for j in range(i):
                        a2 = atom_list[j]
                        if (a1, a2) not in existing and (a2, a1) not in existing:
                            force.addExclusion(a1, a2)
        return system


class IntraMolecularNonBondedForcesModification(BaseModification):
    NAME = "_lambda_intramolecular_nonbonded"

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        reaction_field: bool = False,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Add the intramolecular nonbonded forces as a CustomBondForce to the System.

        Parameters
        ----------
        system : openmm.System
            The System to modify.
        alchemical_atoms : list of int
            The indices of the alchemical atoms.
        reaction_field : bool, optional, default=False
            If True, the energy expression will include a reaction field term.
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        system : openmm.System
            The modified System.

        Notes
        -----
        This code is heavily inspired on REF.
        """
        forces = {force.__class__.__name__: force for force in system.getForces()}
        nb_force = forces["NonbondedForce"]

        if reaction_field:
            # Read: https://github.com/openmm/openmm/issues/3281
            # Read: http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html?highlight=cutoffperiodic
            cutoff = nb_force.getCutoffDistance()
            eps_solvent = nb_force.getReactionFieldDielectric()
            krf = (1 / (cutoff**3)) * (eps_solvent - 1) / (2 * eps_solvent + 1)
            crf = (1 / cutoff) * (3 * eps_solvent) / (2 * eps_solvent + 1)
            energy_expression = "138.9354558466661*charge_prod*(1/r + krf*r*r - crf) + 4*epsilon*((sigma/r)^12-(sigma/r)^6);"
            energy_expression += f"krf = {krf.value_in_unit(_unit.nanometer**-3)};"
            energy_expression += f"crf = {crf.value_in_unit(_unit.nanometer**-1)}"
        else:
            energy_expression = (
                "138.9354558466661*charge_prod/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)"
            )

        internal_nonbonded = _mm.CustomBondForce(energy_expression)
        internal_nonbonded.addPerBondParameter("charge_prod")
        internal_nonbonded.addPerBondParameter("sigma")
        internal_nonbonded.addPerBondParameter("epsilon")
        num_particles = system.getnum_particles()
        atom_charge = [0] * num_particles
        atom_sigma = [0] * num_particles
        atom_epsilon = [0] * num_particles
        for i in range(num_particles):
            charge, sigma, epsilon = nb_force.getParticleParameters(i)
            atom_charge[i] = charge
            atom_sigma[i] = sigma
            atom_epsilon[i] = epsilon
        exceptions = {}
        for i in range(nb_force.getNumExceptions()):
            p1, p2, charge_prod, sigma, epsilon = nb_force.getExceptionParameters(i)
            exceptions[(p1, p2)] = (charge_prod, sigma, epsilon)
        for p1 in alchemical_atoms:
            for p2 in alchemical_atoms:
                if p1 == p2:
                    break
                if (p1, p2) in exceptions:
                    charge_prod, sigma, epsilon = exceptions[(p1, p2)]
                elif (p2, p1) in exceptions:
                    charge_prod, sigma, epsilon = exceptions[(p2, p1)]
                else:
                    charge_prod = atom_charge[p1] * atom_charge[p2]
                    sigma = 0.5 * (atom_sigma[p1] + atom_sigma[p2])
                    epsilon = _unit.sqrt(atom_epsilon[p1] * atom_epsilon[p2])
                if charge_prod._value != 0 or epsilon._value != 0:
                    internal_nonbonded.addBond(p1, p2, [charge_prod, sigma, epsilon])

        system.addForce(internal_nonbonded)

        return system


Alchemist.register_modification_factory(
    IntraMolecularNonBondedExceptionsModification.NAME,
    IntraMolecularNonBondedExceptionsModificationFactory(),
)

Alchemist.register_modification_factory(
    IntraMolecularNonBondedForcesModification.NAME,
    IntraMolecularNonBondedForcesModificationFactory(),
)
