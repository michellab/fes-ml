"""Module for classes related to modifications of the intramolecular terms and their factories."""

import logging
from typing import List

import openmm as _mm
import openmm.unit as _unit

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


class IntraMolecularBondedRemovalModificationFactory(BaseModificationFactory):
    """Factory for creating IntraMolecularBondedRemovalModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of IntraMolecularBondedRemovalModification.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        IntraMolecularBondedRemovalModification
            The modification to be applied.
        """
        return IntraMolecularBondedRemovalModification(*args, **kwargs)


class IntraMolecularNonBondedExceptionsModification(BaseModification):
    """Class to add zeroed exceptions between alchemical atoms."""

    NAME = "IntraMolecularNonBondedExceptions"

    def apply(
        self, system: _mm.System, alchemical_atoms: List[int], *args, **kwargs
    ) -> _mm.System:
        """
        Add exceptions to the NonbondedForce and CustomNonbondedForces.

        Prevents the alchemical atoms from interacting as these interactions
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
    """Class to add intramolecular nonbonded forces as a CustomBondForce."""

    NAME = "IntraMolecularNonBondedForces"

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
            energy_expression = "138.9354558466661*chargeProd*(1/r + krf*r*r - crf) + 4*epsilon*((sigma/r)^12-(sigma/r)^6);"
            energy_expression += f"krf = {krf.value_in_unit(_unit.nanometer**-3)};"
            energy_expression += f"crf = {crf.value_in_unit(_unit.nanometer**-1)}"
        else:
            energy_expression = (
                "138.9354558466661*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)"
            )

        internal_nonbonded = _mm.CustomBondForce(energy_expression)
        internal_nonbonded.addPerBondParameter("chargeProd")
        internal_nonbonded.addPerBondParameter("sigma")
        internal_nonbonded.addPerBondParameter("epsilon")
        num_particles = system.getNumParticles()
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


class IntraMolecularBondedRemovalModification(BaseModification):
    """Class to remove bonded interactions (by zeroing them) between alchemical atoms."""

    NAME = "IntraMolecularBondedRemoval"

    @staticmethod
    def _should_remove(term_atoms: tuple, atom_set: set, remove_in_set: bool) -> bool:
        """
        Determine if a bonded term should be removed based on the atoms it connects.

        Parameters
        ----------
        term_atoms: tuple
            The atoms that the term connects.
        atom_set: set
            The set of atoms to remove the bonded interactions between.
        remove_in_set: bool
            If True, any bonded term connecting atoms in the specified set is removed.  If False,
            any term that does *not* connect atoms in the specified set is removed

        Returns
        -------
        bool
            True if the term should be removed, False otherwise.
        """
        return all(a in atom_set for a in term_atoms) == remove_in_set

    @staticmethod
    def _remove_bonded_interactions(
        force: _mm.Force, alchemical_atoms: List[int], remove_in_set: bool
    ) -> _mm.System:
        """
        Remove all bonded interactions between atoms in a particular set.

        Parameters
        ----------
        force: _mm.Force
            The force to remove the bonded interactions from.
        alchemical_atoms: List[int]
            The indices of the alchemical atoms.
        remove_in_set: bool
            If True, any bonded term connecting atoms in the specified set is removed.  If False,
            any term that does *not* connect atoms in the specified set is removed

        Returns
        -------
        mm.System
            A new System with the specified interactions removed.
        """
        atom_set = set(alchemical_atoms)

        if hasattr(force, "addBond"):
            for i in range(force.getNumBonds()):
                try:
                    # HarmonicBondForce
                    p1, p2, length, k = force.getBondParameters(i)
                    if IntraMolecularBondedRemovalModification._should_remove(
                        (p1, p2), atom_set, remove_in_set
                    ):
                        force.setBondParameters(i, p1, p2, length, 0)
                except ValueError:
                    # CustomBondForce
                    p1, p2, params = force.getBondParameters(i)
                    if IntraMolecularBondedRemovalModification._should_remove(
                        (p1, p2), atom_set, remove_in_set
                    ):
                        zero_params = [1e-9 for _ in params]
                        force.setBondParameters(i, p1, p2, zero_params)
        if hasattr(force, "addAngle"):
            for i in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(i)
                if IntraMolecularBondedRemovalModification._should_remove(
                    (p1, p2, p3), atom_set, remove_in_set
                ):
                    force.setAngleParameters(i, p1, p2, p3, angle, 0)

        if hasattr(force, "addTorsion"):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                if IntraMolecularBondedRemovalModification._should_remove(
                    (p1, p2, p3, p4), atom_set, remove_in_set
                ):
                    force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0)

        return force

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        remove_in_set: bool = True,
        remove_constraints: bool = True,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Remove all bonded interactions between atoms in a particular set.

        Parameters
        ----------
        system: System
            The system to remove the bonded interactions from.
        alchemical_atoms: List[int]
            The indices of the alchemical atoms.
        remove_in_set: bool
            If True, any bonded term connecting atoms in the specified set is removed.  If False,
            any term that does *not* connect atoms in the specified set is removed
        remove_constraints: bool
            If True, remove constraints between pairs of atoms in the set

        Returns
        -------
        mm.System
            A new System with the specified interactions removed.
        """
        # Remove bonded interactions between atoms in the set
        for id, force in enumerate(system.getForces()):
            IntraMolecularBondedRemovalModification._remove_bonded_interactions(
                force, alchemical_atoms, remove_in_set
            )

        if remove_constraints:
            for i in range(system.getNumConstraints(), 0, -1):
                p1, p2, length = system.getConstraintParameters(i - 1)
                if IntraMolecularBondedRemovalModification._should_remove(
                    (p1, p2), set(alchemical_atoms), remove_in_set
                ):
                    system.removeConstraint(i - 1)

        return system
