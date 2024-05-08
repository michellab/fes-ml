"""Module for the EMLEPotentialModification class and its factory."""
from typing import List

import numpy as _np
import openmm as _mm
import sire as _sr
from emle.calculator import EMLECalculator as _EMLECalculator

from .base_modification import BaseModification, BaseModificationFactory


class EMLEPotentialModificationFactory(BaseModificationFactory):
    """Factory for creating EMLEPotentialModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of EMLEPotential.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        EMLEPotential
            The modification to be applied.
        """
        return EMLEPotentialModification(*args, **kwargs)


class EMLEPotentialModification(BaseModification):
    """Modification to add the EMLE potential to the OpenMM System."""

    NAME = "EMLEPotential"
    pre_dependencies = None
    post_dependencies = None

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        lambda_value: float,
        mols: _sr.mol.Molecule,
        parm7: str,
        mm_charges: _np.ndarray,
        backend: str = None,
        method: str = "electrostatic",
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Add the EMLE potential to the OpenMM system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System to modify.
        alchemical_atoms : list of int
            The indices of the alchemical atoms in the System.
        lambda_value : float
            The value of the alchemical state parameter.
        mols : sire.mol.Molecule
            The molecular system.
        parm7 : str
            The path to the topology file.
        mm_charges : np.ndarray
            The charges of the atoms in the ML region.
        args : list
            Additional arguments to be passed to the EMLE calculator.
        backend : str, optional, default=None
            The backend to use for the EMLE calculation.
        method : str, optional, default="electrostatic"
            The method to use for the EMLE calculation.
        kwargs : dict
            Additional keyword arguments to be passed to the EMLECalculator.
            See https://github.com/chemle/emle-engine/blob/main/emle/calculator.py#L399-L519.

        Returns
        -------
        system : openmm.System
            The modified OpenMM System with the EMLE potential added.
        """
        # Create a calculator.
        calculator = _EMLECalculator(
            backend=backend,
            method="mm",
            parm7=parm7,
            mm_charges=mm_charges,
            lambda_interpolate=lambda_value,
            qm_indices=alchemical_atoms,
            *args,
            **kwargs,
        )

        # Create a perturbable molecular system and EMLEEngine. (First molecule is QM region.)
        mols, engine = _sr.qm.emle(
            mols, mols.atoms(alchemical_atoms), calculator, "12A", 20
        )

        # Get the OpenMM EMLEForce object.
        emle_force, _ = engine.get_forces()

        system.addForce(emle_force)
        # Zero the charges on the atoms within the QM region
        for force in system.getForces():
            if isinstance(force, _mm.NonbondedForce):
                for i in range(force.getNumParticles()):
                    if i in alchemical_atoms:
                        _, sigma, epsilon = force.getParticleParameters(i)
                        force.setParticleParameters(i, 1e-9, sigma, epsilon)

        return system
