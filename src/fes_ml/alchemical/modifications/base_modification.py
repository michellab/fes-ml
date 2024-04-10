"""Base classes for alchemical modifications."""

import openmm as _mm


class BaseModification:
    """Base class for classes that define alchemical modifications.

    If you are defining a new alchemical modification, you need to create a subclass of
    BaseModification and BaseModificationFactory. When Alchemist.apply_modifications is called,
    it looks up the factories that have been registered with the name of the lambda values to
    be applied, and uses those factories to create BaseModification(s) of the appropriate subclass.

    The apply method must be implement in the BaseModification subclass to define the alchemical
    modification to be applied to the system.
    """

    def apply(
        self,
        *args,
        **kwargs,
    ) -> _mm.System:
        """Apply the modification to the system.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        openmm.System
            The modified system.
        """
        raise NotImplementedError(
            "Apply method must be implemented for the modification."
        )


class BaseModificationFactory:
    """Base class for classes that define alchemical modification factories.

    If you are defining a new alchemical modification, you need to create a subclass of
    BaseModification and BaseModificationFactory, and register an instance of the factory
    by calling Alchemist.register_modification_factory.
    """

    def create_modification(self, *args, **kwargs) -> BaseModification:
        """Create an instance of the modification.

        Parameters
        ----------
        args : list
            Additional arguments to be passed to the modification.
        kwargs : dict
            Additional keyword arguments to be passed to the modification.

        Returns
        -------
        BaseModification
            The modification to be applied.
        """
        raise NotImplementedError("create_modification method must be implemented")
