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
      add B to A's ``skip_dependencies`` class-level attribute.
    - If multiple modifications imply B, B is applied only once.
    - Modifications are applied in topologically sorted order based on
      the dependency graph.
    - A modification must have a default key controlled by the class-level
      attribute ``NAME``.
    - A modication can also have a custom name by passing a string
      ``modification_name`` when instantiating the modification. This
      allows multiple instances of the same modification to coexist in
      the same system, each with its own parameters.
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

    def __init__(self, modification_name: str = None):
        """
        Initialize the BaseModification.

        Parameters
        ----------
        modification_name : str, optional
            Custom name for this modification instance. If not provided,
            uses the class NAME.
        """
        assert modification_name.count(":") <= 1, (
            f"Invalid modification_name '{modification_name}': it may contain at most one ':' to indicate the alchemical group."
        )
        self.modification_name = modification_name or self.NAME

    def __init_subclass__(cls, **kwargs):
        """
        Initialize the subclass.

        Force subclasses to override the NAME attribute.
        """
        super().__init_subclass__(**kwargs)
        if cls.NAME is NotImplemented:
            raise NotImplementedError(f"Modification {cls.__name__} must override 'NAME' class-level attribute.")

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

    @staticmethod
    def find_forces_by_group(system: _mm.System, group: str) -> List[_mm.Force]:
        """
        Find all forces in the system that belong to a given alchemical group.

        Notes
        -----
        Alchemical groups are defined by suffixes in the force names, e.g., ':region1'.

        Parameters
        ----------
        system : openmm.System
            The OpenMM system to search.
        group : str
            The alchemical group to match (e.g., ':region1').

        Returns
        -------
        List[openmm.Force]
            List of forces whose names end with the specified alchemical group.
        """
        return [force for force in system.getForces() if force.getName().endswith(f":{group}")]

    @staticmethod
    def find_force_by_name(system: _mm.System, force_name: str) -> _mm.Force:
        """
        Find a force in the system by its full name.

        Parameters
        ----------
        system : openmm.System
            The OpenMM system to search.
        force_name : str
            The name of the force to find.

        Returns
        -------
        openmm.Force
            The force with the specified name.

        Raises
        ------
        ValueError
            If no force with the specified name is found.
        """
        for force in system.getForces():
            if force.getName() == force_name:
                return force
        raise ValueError(f"Force with name '{force_name}' not found in system.")

    @property
    def alchemical_group(self) -> str:
        """
        Get the suffix of the current instance name.

        Returns
        -------
        str
            The suffix part after ':' if present, empty string otherwise.
        """
        if ":" in self.modification_name:
            return self.modification_name.split(":", 1)[1]
        return ""


class BaseModificationFactory(ABC):
    """
    Base class for classes that define modification factories.

    To create a new modification, create subclasses of ``BaseModification`` and
    ``BaseModificationFactory``.
    """

    @abstractmethod
    def create_modification(self, modification_name: str = None, *args, **kwargs) -> BaseModification:
        """
        Create an instance of the modification.

        Parameters
        ----------
        modification_name : str, optional
            Custom name for this modification instance.

        Returns
        -------
        BaseModification
            Instance of the modification to be applied.
        """
        pass
