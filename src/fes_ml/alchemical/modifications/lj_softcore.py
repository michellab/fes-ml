"""Module for the LJSoftCoreModification class and its factory."""

import logging
from typing import List, Optional, Union

import openmm as _mm
import openmm.app as _app

from .base_modification import BaseModification, BaseModificationFactory
from .intramolecular import (
    IntraMolecularNonBondedExceptionsModification,
    IntraMolecularNonBondedForcesModification,
)

logger = logging.getLogger(__name__)


class LJSoftCoreModificationFactory(BaseModificationFactory):
    """Factory for creating LJSoftCoreModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """
        Create an instance of LJSoftCoreModification.

        Returns
        -------
        LJSoftCoreModification
            Instance of the modification to be applied.
        """
        return LJSoftCoreModification(*args, **kwargs)


class LJSoftCoreModification(BaseModification):
    """Class to add a softcore Lennard-Jones potential to the System."""

    NAME = "LJSoftCore"

    _NON_BONDED_METHODS = {
        0: _app.NoCutoff,
        1: _app.CutoffNonPeriodic,
        2: _app.CutoffPeriodic,
        3: _app.Ewald,
        4: _app.PME,
    }

    pre_dependencies: List[str] = [IntraMolecularNonBondedForcesModification.NAME]
    post_dependencies: List[str] = [IntraMolecularNonBondedExceptionsModification.NAME]

    def apply(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        lambda_value: Optional[Union[float, int]] = 1.0,
        include_repulsion: bool = True,
        include_attraction: bool = True,
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
        lambda_value : float
            The value of the alchemical state parameter.
        include_repulsion : bool, optional
            Whether to include the short-range repulsive term (r^-12 component).
            Default is True.
        include_attraction : bool, optional
            Whether to include the long-range attractive term (r^-6 component).
            Default is True.
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
        nb_force = forces["NonbondedForce"]

        if not include_repulsion and not include_attraction:
            raise ValueError("At least one of include_repulsion or include_attraction must be True.")

        # Define the softcore Lennard-Jones energy function
        # Standard LJ: 4*epsilon*(x^2 - x) where x = (sigma/r)^6
        # Softcore LJ: 4*epsilon*(x_soft^2 - x_soft) where x_soft = (sigma/reff)^6
        terms = []
        # Main energy expression pieces
        if include_repulsion:
            terms.append(f"{lambda_value}*4*epsilon*x*x")
        if include_attraction:
            terms.append(f"-{lambda_value}*4*epsilon*x*fdamp")

        energy_expr = " + ".join(terms)

        # Full OpenMM expression: E; intermediate definitions follow
        expr = (
            f"{energy_expr}; "
            "x=(sigma/reff)^6; "
            f"reff=sigma*(0.5*(1.0-{lambda_value})+(r/sigma)^6)^(1/6); "
            "sigma=0.5*(sigma1+sigma2); "
            "epsilon=sqrt(epsilon1*epsilon2); "
        )

        # Add damping if needed
        if include_attraction and not include_repulsion:
            expr += (
                # Tang-Toennies f6(b*r) expansion, truncated to 6th order
                "fdamp=1.0 - exp(-xdamp)*(1.0 + xdamp + 0.5*xdamp^2 + "
                "(1.0/6.0)*xdamp^3 + (1.0/24.0)*xdamp^4 + "
                "(1.0/120.0)*xdamp^5 + (1.0/720.0)*xdamp^6);"
                "xdamp=bdamp*r; "
                "bdamp=1/0.021; "
            )
        else:
            expr += "fdamp=1.0;"

        """
        energy_function = 'lambda*4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;'
        energy_function += 'reff_sterics = sigma*(0.5*(1.0-lambda) + (r/sigma)^6)^(1/6);'
        energy_function += 'sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);'
        custom_force = CustomNonbondedForce(energy_function)
        """

        logger.debug(f"Softcore LJ expression: {expr}")

        # Create a CustomNonbondedForce to compute the softcore Lennard-Jones
        soft_core_force = _mm.CustomNonbondedForce(expr)

        if self._NON_BONDED_METHODS[nb_force.getNonbondedMethod()] in [
            self._NON_BONDED_METHODS[3],
            self._NON_BONDED_METHODS[4],
        ]:
            logger.warning("The softcore Lennard-Jones interactions are not implemented for Ewald or PME")
            logger.warning("The nonbonded method will be set to CutoffPeriodic")
            soft_core_force.setNonbondedMethod(2)
        else:
            soft_core_force.setNonbondedMethod(nb_force.getNonbondedMethod())

        soft_core_force.setCutoffDistance(nb_force.getCutoffDistance())
        soft_core_force.setUseSwitchingFunction(nb_force.getUseSwitchingFunction())
        soft_core_force.setSwitchingDistance(nb_force.getSwitchingDistance())

        if abs(lambda_value) < 1e-8:
            # Cannot use long range correction with a force that does not depend on r
            soft_core_force.setUseLongRangeCorrection(False)
        else:
            soft_core_force.setUseLongRangeCorrection(nb_force.getUseDispersionCorrection())

        # https://github.com/openmm/openmm/issues/1877
        # Set the values of sigma and epsilon by copying them from the existing NonBondedForce
        # Epsilon will always be 0 for the ML atoms as the LJ 12-6 interaction is removed
        soft_core_force.addPerParticleParameter("sigma")
        soft_core_force.addPerParticleParameter("epsilon")
        for index in range(system.getNumParticles()):
            [charge, sigma, epsilon] = nb_force.getParticleParameters(index)
            soft_core_force.addParticle([sigma, epsilon])
            if index in alchemical_atoms:
                # Remove the LJ 12-6 interaction (if 1e-9, the free energies are weirdly wrong)
                nb_force.setParticleParameters(index, charge, sigma, 0.0)

        # Set the custom force to occur between just the alchemical particle and the other particles
        mm_atoms = set(range(system.getNumParticles())) - set(alchemical_atoms)
        soft_core_force.addInteractionGroup(alchemical_atoms, mm_atoms)

        # Add the CustomNonbondedForce to the System
        soft_core_force.setName(self.modification_name)
        system.addForce(soft_core_force)

        return system
