"""Module for the EMLEPotentialModification class and its factory."""

import logging
from typing import List, Optional

import numpy as _np
import openmm as _mm
import sire as _sr

from .base_modification import BaseModification, BaseModificationFactory

logger = logging.getLogger(__name__)

_EMLE_CALCULATORS = []


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

    pre_dependencies = ["IntraMolecularNonBondedForces"]
    post_dependencies = ["IntraMolecularNonBondedExceptions"]

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        lambda_value: float,
        mols: _sr.mol.Molecule,
        parm7: str,
        mm_charges: _np.ndarray,
        openmm_charges: List[float] = None,
        backend: str = None,
        method: str = "electrostatic",
        torch_model: bool = False,
        cutoff: str = "12A",
        neighbour_list_frequency: int = 0,
        device: Optional[str] = None,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Add the EMLE potential to the OpenMM system.

        Note that the charges of the atoms in the alchemical region are set to zero,
        which may interfere with other modifications.

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
        torch_model : bool, optional, default=False
            Whether to use a PyTorch model for the EMLE calculation.
            Requires the feature_aev branch of the emle package.
        cutoff : str, optional, default="12A"
            The cutoff to use for the EMLE calculation.
        neighbour_list_frequency : int, optional, default=20
            The frequency at which to update the neighbour list.
        device : str, optional, default=None
            The device to use for the EMLE calculation.
        kwargs : dict
            Additional keyword arguments to be passed to the EMLECalculator or EMLE Torch model.""
            See https://github.com/chemle/emle-engine/blob/main/emle/calculator.py#L399-L519.

        Returns
        -------
        system : openmm.System
            The modified OpenMM System with the EMLE potential added.
        """
        from emle.calculator import EMLECalculator as _EMLECalculator

        if lambda_value is None:
            # This can only occur when MLInterpolation is also being used.
            # There must be a lambda_interpolate global variable available.
            # What we do is to find the lambda_interpolate value from the MLInterpolation force,
            # and use that to create the EMLECalculator.
            # Because we also add the interpolation force, we are sure that the lambda_interpolate
            # does not scale the EMLE potential.
            cv_force = [f for f in system.getForces() if f.getName() == "MLInterpolation"]
            assert len(cv_force) == 1, f"There are {len(cv_force)} MLInterpolation forces. Only one is allowed."

            cv_force = cv_force[0]
            assert isinstance(cv_force, _mm.CustomCVForce), f"Expected a CustomCVForce, but got {type(cv_force)}."

            for i in range(cv_force.getNumGlobalParameters()):
                if cv_force.getGlobalParameterName(i) == "lambda_interpolate":
                    lambda_value = cv_force.getGlobalParameterDefaultValue(i)
                    break

            logger.debug(f"Using λ={lambda_value} to create EMLE potential. This is the lambda_interpolate value of MLInterpolation force.")

            if backend is not None:
                logger.warning(
                    f"The EMLE potential backend is {backend}, but an MLInterpolation force is present. "
                    "Typically, backend should be None when using MLInterpolation. "
                    "Make sure this is the desired behavior."
                )

        if backend is not None and self.post_dependencies is None:
            logger.warning(
                f"The EMLE potential backend is {backend} but there are no post-dependencies. "
                "This may cause the EMLE potential to not be applied correctly as the MM intramolecular "
                "terms may not have been zeroed out in the OpenMM System. "
                "Make sure this is the desired behavior."
            )

        if torch_model:
            try:
                from emle.models import EMLE as _EMLE

                calculator = _EMLE(*args, **kwargs).to(device)
            except ImportError:
                raise ImportError("The feature_aev branch of the emle package is required when using an EMLE torch model.")
        else:
            # Create a calculator.
            calculator = _EMLECalculator(
                backend=backend,
                method=method,
                parm7=parm7,
                mm_charges=mm_charges,
                lambda_interpolate=lambda_value,
                qm_indices=alchemical_atoms,
                device=device,
                *args,
                **kwargs,
            )
            _EMLE_CALCULATORS.append(calculator)

        # Create a perturbable molecular system and EMLEEngine. (First molecule is QM region.)
        mols, engine = _sr.qm.emle(
            mols,
            mols.atoms(alchemical_atoms),
            calculator,
            cutoff,
            neighbour_list_frequency,
        )

        # Set the system charges explicitly if they are provided.
        if openmm_charges is not None:
            engine.set_charges(openmm_charges)

        # Get the OpenMM EMLEForce object.
        emle_force, interpolation_force = engine.get_forces()

        # Add the EMLE force to the system.
        emle_force.setName(self.modification_name)
        system.addForce(emle_force)
        # Add the interpolation force to the system, so that the EMLE force does not scale
        # with the MLInterpolation force.
        interpolation_force.setName("EMLECustomBondForce_" + self.modification_name)
        interpolation_force.setEnergyFunction("0.0")  # No energy contribution.
        system.addForce(interpolation_force)

        # Determine if LJ parameters should be zeroed out.
        disp_mode = kwargs.get("dispersion_mode", None)
        if disp_mode not in (None, "lj", "c6"):
            logger.warning(f"dispersion_mode {disp_mode} not recognised. Must be one of None, 'lj', or 'c6'. ")
        zero_lj = disp_mode in ("lj", "c6")

        # Zero the charges and/or LJ parameters on the atoms within the QM region
        logger.debug(f"Zeroing charges and{' LJ parameters' if zero_lj else ''} for atoms in QM region: {alchemical_atoms}")
        for force in system.getForces():
            if isinstance(force, _mm.NonbondedForce):
                for i in alchemical_atoms:
                    try:
                        _, sigma, epsilon = force.getParticleParameters(i)
                        force.setParticleParameters(i, 0, sigma, epsilon if not zero_lj else 0)
                    except _mm.OpenMMException:
                        logger.warning(f"Could not set charge and/or LJ parameters to 0 for atom {i}. Check if this is the expected behaviour.")

        return system
