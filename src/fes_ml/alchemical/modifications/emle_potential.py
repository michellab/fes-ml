import openmm as _mm

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
    NAME = "EMLEPotential"
    pre_dependencies = None
    post_dependencies = None

    def apply(self, system: _mm.System, *args, **kwargs) -> _mm.System:
        pass
