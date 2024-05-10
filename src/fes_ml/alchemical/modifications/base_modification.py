"""Base classes for system modifications."""

from abc import ABC, abstractmethod
from typing import List

import openmm as _mm


class BaseModification(ABC):
    """
    Base class for defining modifications to an OpenMM system.

    Specifications for modifications in fes-ml:

    - A modification alters the base OpenMM system.
    - A modification can be independent or dependent on other modifications.
    - If modification A implies B, add B to A's ``post_dependencies``
      class-level attribute.
    - If modification A requires B, add B to A's ``pre_dependencies``
      class-level attribute.
    - If multiple modifications imply B, B is applied only once.
    - Modification are applied in topologically sorted order based on
      the dependency graph.
    - A modification must an associated key controlled by the class-level
      attribute ``NAME``.
    - Modifications can be applied by passing a lambda schedule
      dictionary when creating alchemical states. This dictionary
      maps ``NAME``s to λ values for each ``AlchemicalState``.

    Implementing new modifications:

    - To define a new modification, create a subclass of ``BaseModification``
      and ``BaseModificationFactory``.
    - Every ``BaseModification`` subclass must implement the apply method to
    define the modification on the OpenMM system and override the ``NAME``
    class-level attribute.
    """

    NAME: str = NotImplemented
    pre_dependencies: List[str] = []
    post_dependencies: List[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.NAME is NotImplemented:
            raise NotImplementedError(
                f"Modification {cls.__name__} must override 'NAME' class-level attribute."
            )

    @abstractmethod
    def apply(
        self,
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Apply the modification to the system.

        Returns
        -------
        openmm.System
            The modified system.
        """
        pass


class BaseModificationFactory(ABC):
    """
    Base class for classes that define modification factories.

    To create a new modification, create subclasses of ``BaseModification`` and
    ``BaseModificationFactory``.
    """

    @abstractmethod
    def create_modification(self, *args, **kwargs) -> BaseModification:
        """
        Create an instance of the modification.

        Returns
        -------
        BaseModification
            Instance of the modification to be applied.
        """
        pass
