"""Module for the Alchemist class."""
from typing import Dict

import openmm as _mm


from typing import List
import logging 

from .modifications.base_modification import BaseModificationFactory
from .modifications.lj_modification import LJSoftCoreModification
from .modifications.charges_modification import ChargesModification
from .modifications.ml_modification import MLModification
from .modifications.ml_corr_modification import MLCorrectionModification
from .modifications.intramolecular_modification import IntraMolecularNonBondedExceptionsModification, IntraMolecularNonBondedForcesModification

logger = logging.getLogger(__name__)

class Alchemist:
    _modification_factories: Dict[str, BaseModificationFactory] = {}

    @staticmethod
    def register_modification_factory(
        name: str, factory: BaseModificationFactory
    ) -> Dict[str, BaseModificationFactory]:
        Alchemist._modification_factories[name] = factory

        return Alchemist._modification_factories

    def apply_modifications(
        self, 
        system: _mm.System, 
        alchemical_atoms: List[int],
        lambda_values: Dict[str, float],
        *args, **kwargs
    ) -> _mm.System:
        """
        Apply the alchemical modifications to the system.

        Parameters
        ----------
        system : openmm.System
            The system to be modified.
        lambda_values : dict
            A dictionary of lambda values to be applied to the system.
        args : list
            Additional arguments to be passed to the modifications.
        kwargs : dict
            Additional keyword arguments to be passed to the modifications.

        Returns
        -------
        openmm.System
            The modified system.
        """
        if lambda_values.get(MLModification.NAME, None) and lambda_values.get(MLCorrectionModification.NAME, None):
            # Cannot apply ML and MLCorrection at the same time
            raise ValueError(
                f"Cannot apply {MLModification.NAME} and {MLCorrectionModification.NAME} at the same time."
            )
        
        # Create a list of modifications to apply
        modifications_to_apply = []
        for name, lambda_value in lambda_values.items():
            if name in Alchemist._modification_factories:
                factory = self._modification_factories[name]
                modification = factory.create_modification()
                modifications_to_apply.append((modification, lambda_value))
        
        # If LJSoftCore and/or Charges are applied, and ML is not applied, add intramolecular modifications
        if any(
            [lambda_values.get(ChargesModification.NAME, None),
            lambda_values.get(LJSoftCoreModification.NAME, None)]
            ) and not lambda_values.get(MLModification.NAME, None):
            logger.info("Because LJSoftCore and/or Charges are applied and ML is not applied, adding intramolecular modifications.")
            # Apply intramolecular non-bonded forces modification at the same time
            factory = self._modification_factories[IntraMolecularNonBondedForcesModification.NAME]
            modification = factory.create_modification()
            modifications_to_apply.insert(0, (modification, None))

            # Apply intramolecular non-bonded exceptions modification at the end
            factory = self._modification_factories[IntraMolecularNonBondedExceptionsModification.NAME]
            modification = factory.create_modification(*args, **kwargs)
            modifications_to_apply.append((modification, None))

        
        for modification, lambda_value in modifications_to_apply:
            system = modification.apply(system, lambda_value, alchemical_atoms, *args, **kwargs)

        return system
