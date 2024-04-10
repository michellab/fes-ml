"""This module contains the implementation of the LJSoftCoreModification class."""

import logging
from typing import List

import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit

from ..alchemist import Alchemist
from .base_modification import BaseModification, BaseModificationFactory

logger = logging.getLogger(__name__)


class LJSoftCoreModificationFactory(BaseModificationFactory):
    """Factory for creating LJSoftCoreModification instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of LJSoftCoreModification.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        LJSoftCoreModification
            The modification to be applied.
        """
        return LJSoftCoreModification(*args, **kwargs)


class LJSoftCoreModification(BaseModification):
    _NON_BONDED_METHODS = {
        0: _app.NoCutoff,
        1: _app.CutoffNonPeriodic,
        2: _app.CutoffPeriodic,
        3: _app.Ewald,
        4: _app.PME,
    }

    NAME = "lambda_lj"

    def apply(
        self,
        system: _mm.System,
        lambda_value: float,
        alchemical_atoms: List[int],
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Apply the LJ soft core modification to the system.

        Parameters
        ----------
        system : openmm.System
            The system to be modified.
        lambda_value : float
            The value of the alchemical state parameter.
        alchemical_atoms : list of int
            The indices of the alchemical atoms in the system.
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        openmm.System
            The modified system.
        """
        # Apply the LJ soft core modification
        logger.info(
            f"Applying LJ soft core modification with lambda value: {lambda_value}"
        )

        forces = {force.__class__.__name__: force for force in system.getForces()}
        nb_force = forces["NonbondedForce"]

        # Define the softcore Lennard-Jones energy function
        energy_function = (
            f"{lambda_value}*4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
        )
        energy_function += (
            f"reff_sterics = sigma*(0.5*(1.0-{lambda_value}) + (r/sigma)^6)^(1/6);"
        )
        energy_function += (
            "sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);"
        )

        logger.debug(f"LJ softcore function: {energy_function}")
        # Create a CustomNonbondedForce to compute the softcore Lennard-Jones and Coulomb interactions
        soft_core_force = _mm.CustomNonbondedForce(energy_function)

        if self._NON_BONDED_METHODS[nb_force.getNonbondedMethod()] in [
            self._NON_BONDED_METHODS[3],
            self._NON_BONDED_METHODS[4],
        ]:
            logger.warning(
                "The softcore Lennard-Jones interactions are not implemented for Ewald or PME"
            )
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
            soft_core_force.setUseLongRangeCorrection(
                nb_force.getUseDispersionCorrection()
            )

        # https://github.com/openmm/openmm/issues/1877
        # Set the values of sigma and epsilon by copying them from the existing NonBondedForce
        # Epsilon will always be 0 for the ML atoms as the LJ 12-6 interaction is removed
        soft_core_force.addPerParticleParameter("sigma")
        soft_core_force.addPerParticleParameter("epsilon")
        for index in range(system.getNumParticles()):
            [charge, sigma, epsilon] = nb_force.getParticleParameters(index)
            soft_core_force.addParticle([sigma, epsilon])
            if index in alchemical_atoms:
                # Remove the LJ 12-6 interaction
                nb_force.setParticleParameters(index, charge, sigma, 1e-9)

        # Set the custom force to occur between just the alchemical particle and the other particles
        mm_atoms = set(range(system.getNumParticles())) - set(alchemical_atoms)
        soft_core_force.addInteractionGroup(alchemical_atoms, mm_atoms)

        # Add the CustomNonbondedForce to the System
        system.addForce(soft_core_force)

        return system


# Register the LJSoftCoreModificationFactory with the Alchemist
Alchemist.register_modification_factory(
    LJSoftCoreModification.NAME, LJSoftCoreModificationFactory()
)
