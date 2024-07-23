"""Module for the BaseModification abstract class and its factory."""

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
    - If modification A is incompatible with B being present in the graph,
      add B to A's ``skip_depencies`` class-level attribute.
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
    pre_dependencies: List[str] = None
    post_dependencies: List[str] = None
    skip_dependencies: List[str] = None

    def __init_subclass__(cls, **kwargs):
        """
        Initialize the subclass.

        Force subclasses to override the NAME attribute.
        """
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

    @classmethod
    def add_pre_dependency(cls, name: str, beggining: bool = False) -> None:
        """
        Add a pre-dependency to the modification.

        Parameters
        ----------
        name : str
            Name of the modification to add as a pre-dependency.
        beggining : bool, optional
            Add the pre-dependency to the beginning of the list.
        """
        if beggining:
            cls.pre_dependencies.insert(0, name)
        else:
            cls.pre_dependencies.append(name)

    @classmethod
    def remove_pre_dependency(cls, name: str) -> None:
        """
        Remove a pre-dependency from the modification.

        Parameters
        ----------
        name : str
            Name of the modification to remove as a pre-dependency.
        """
        if name in cls.pre_dependencies:
            cls.pre_dependencies.remove(name)
        else:
            raise ValueError(f"{name} is not a pre-dependency of {cls.NAME}.")

    @classmethod
    def add_post_dependency(cls, name: str, beggining: bool = False) -> None:
        """
        Add a post-dependency to the modification.

        Parameters
        ----------
        name : str
            Name of the modification to add as a post-dependency.
        beggining : bool, optional
            Add the post-dependency to the beginning of the list.
        """
        if beggining:
            cls.post_dependencies.insert(0, name)
        else:
            cls.post_dependencies.append(name)

    @classmethod
    def remove_post_dependency(cls, name: str) -> None:
        """
        Remove a post-dependency from the modification.

        Parameters
        ----------
        name : str
            Name of the modification to remove as a post-dependency.
        """
        if name in cls.post_dependencies:
            cls.post_dependencies.remove(name)
        else:
            raise ValueError(f"{name} is not a post-dependency of {cls.NAME}.")


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
