
import openmm as _mm

from .base_modification import BaseModification, BaseModificationFactory


class MLCorrectionModificationFactory(BaseModificationFactory):
    """Factory for creating MLInterpolate instances."""

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of MLCorrection.

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
        return MLCorrectionModification(*args, **kwargs)


class MLCorrectionModification(BaseModification):
    NAME = "MLCorrection"
    PRE_DEPENDENCIES = ["MLPotential"]
    POST_DEPENDENCIES = ["IntraMolecularNonBondedExceptions", "IntraMolecularBondedRemoval"]

    def apply(self, system: _mm.System, *args, **kwargs) -> _mm.System:
        pass