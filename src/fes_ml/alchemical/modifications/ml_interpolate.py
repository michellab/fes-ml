from typing import List, Optional

import openmm as _mm

from .base_modification import BaseModification, BaseModificationFactory


class MLInterpolateFacotry(BaseModificationFactory):
    """Factory for creating MLInterpolate instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of MLInterpolate.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        MLInterpolate
            The modification to be applied.
        """
        return MLInterpolate(*args, **kwargs)


class MLInterpolate(BaseModification):
    NAME = "MLInterpolate"
    PRE_DEPENDENCIES = ["MLPotential"]
    POST_DEPENDENCIES = ["IntraMolecularNonBondedExceptions", "IntraMolecularBondedRemoval"]

    def apply(self, system: _mm.System, *args, **kwargs) -> _mm.System:
        pass