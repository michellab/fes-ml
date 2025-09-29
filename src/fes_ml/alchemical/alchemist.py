"""Module for the Alchemist class."""

import logging
from copy import deepcopy as _deepcopy
from importlib.metadata import entry_points
from typing import Any, Dict, List, Optional

import networkx as nx
import openmm as _mm

from .modifications.base_modification import BaseModification, BaseModificationFactory

logger = logging.getLogger(__name__)


class Alchemist:
    """A class for applying alchemical modifications to an OpenMM system."""

    _modification_factories: Dict[str, BaseModificationFactory] = {}
    _DEFAULT_ALCHEMICAL_GROUP = ":default"

    @staticmethod
    def register_modification_factory(
        name: str, factory: BaseModificationFactory
    ) -> Dict[str, BaseModificationFactory]:
        """Register a new modification factory in the Alchemist class."""
        Alchemist._modification_factories[name] = factory

        return Alchemist._modification_factories

    def __init__(self) -> None:
        """Initialize the Alchemist object."""
        logger.debug("-" * 100)
        logger.debug("⌬ ALCHEMIST ⌬")
        logger.debug("-" * 100)

        self._graph = nx.DiGraph()

    def __del__(self) -> None:
        """Delete the Alchemist object."""
        logger.debug("-" * 100)

    def __repr__(self) -> str:
        """Return the string representation of the Alchemist object."""
        return nx.to_dict_of_lists(self._graph)

    @property
    def graph(self) -> nx.DiGraph:
        """Set the graph of alchemical modifications."""
        return self._graph

    def add_modification(self, name: str, factory: BaseModificationFactory):
        """
        Add a modification to the Alchemist graph.

        Parameters
        ----------
        name : str
            The name of the modification.
        factory : BaseModificationFactory
            The factory to create the modification.
        """
        self._modification_factories[name] = factory

    def plot_graph(self):
        """Plot the graph of the alchemical modifications."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        nx.draw(
            self._graph,
            pos=nx.circular_layout(self._graph),
            with_labels=True,
            node_color="skyblue",
            node_size=10000,
            edge_color="gray",
        )
        plt.title("Alchemical Graph")
        plt.savefig("alchemical_graph.png")
        plt.close()

    def reset_alchemical_graph(self):
        """Reset the graph of alchemical modifications."""
        self._graph = nx.DiGraph()

    def add_modification_to_graph(
        self, modification: BaseModification, lambda_value: Optional[float] = 1.0
    ) -> None:
        """
        Add a modification to the graph of alchemical modifications.

        Parameters
        ----------
        modification : BaseModification
            The modification to add to the graph.
        lambda_value : float
            The value of the alchemical state parameter.
        """
        node_name = modification.modification_name

        if node_name in self._graph.nodes and lambda_value is None:
            lambda_value = self._graph.nodes[node_name].get("lambda_value", None)

        self._graph.add_node(
            node_name, modification=modification, lambda_value=lambda_value
        )

        if modification.pre_dependencies is not None:
            for pre_dependency in modification.pre_dependencies:
                if pre_dependency not in self._modification_factories:
                    raise ValueError(
                        f"Pre-dependency {pre_dependency} of {modification.NAME} "
                        "not found in the factories. Please make sure there are no "
                        "typos in the name of this pre-dependency and that the target "
                        "modification is implemented and registered as an entry point."
                    )

                # Check if dependency already exists in graph
                dep_modification_name = (
                    f"{pre_dependency}:{modification.alchemical_group}"
                )
                if dep_modification_name not in self._graph.nodes:
                    factory = self._modification_factories[pre_dependency]
                    pre_modification = factory.create_modification(
                        modification_name=dep_modification_name
                    )
                    self.add_modification_to_graph(pre_modification, None)

                self._graph.add_edge(dep_modification_name, node_name)

        if modification.post_dependencies is not None:
            for post_dependency in modification.post_dependencies:
                if post_dependency not in self._modification_factories:
                    raise ValueError(
                        f"Post-dependency {post_dependency} of {modification.NAME} "
                        "not found in the factories. Please make sure there are no "
                        "typos in the name of this post-dependency and that the target "
                        "modification is implemented and registered as an entry point."
                    )

                # Check if dependency already exists in graph
                dep_modification_name = (
                    f"{post_dependency}:{modification.alchemical_group}"
                )
                if dep_modification_name not in self._graph.nodes:
                    factory = self._modification_factories[post_dependency]
                    post_modification = factory.create_modification(
                        modification_name=dep_modification_name
                    )
                    self.add_modification_to_graph(post_modification, None)

                self._graph.add_edge(node_name, dep_modification_name)

    def remove_modification_from_graph(self, modification: str) -> None:
        """
        Remove modification from the graph of alchemical modifications.

        Parameters
        ----------
        modification
            Name of the node to remove
        """
        self._graph.remove_node(modification)

    def create_alchemical_graph(
        self,
        lambda_schedule: Dict[str, float],
        additional_modifications: Optional[List[str]] = None,
        modifications_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Create a graph of alchemical modifications to apply.

        Parameters
        ----------
        lambda_schedule : dict
            A dictionary of λ values to be applied to the system.
        additional_modifications : list of str
            Additional modifications to apply.
        modifications_kwargs : dict
            A dictionary of keyword arguments for the modifications.

        Returns
        -------
        nx.DiGraph
            The graph of the modifications to apply.
        """
        logger.debug("Creating graph of alchemical modifications.")
        for name, lambda_value in lambda_schedule.items():
            if ":" in name:
                base_name, modification_name = name.split(":", 1)[0], name
            else:
                base_name, modification_name = (
                    name,
                    name + self._DEFAULT_ALCHEMICAL_GROUP,
                )

            if modifications_kwargs is not None and name in modifications_kwargs:
                if modification_name not in modifications_kwargs:
                    modifications_kwargs[modification_name] = modifications_kwargs.pop(
                        name
                    )

            if base_name in Alchemist._modification_factories:
                factory = self._modification_factories[base_name]
                modification = factory.create_modification(
                    modification_name=modification_name
                )
                self.add_modification_to_graph(modification, lambda_value=lambda_value)
            else:
                raise ValueError(
                    f"Modification {base_name} not found in the factories."
                )

        if additional_modifications is not None:
            for name in additional_modifications:
                if ":" in name:
                    base_name, modification_name = name.split(":", 1)[0], name
                else:
                    base_name, modification_name = (
                        name,
                        name + self._DEFAULT_ALCHEMICAL_GROUP,
                    )

                if modifications_kwargs is not None and name in modifications_kwargs:
                    if modification_name in modifications_kwargs:
                        raise ValueError(
                            f"Cannot rename '{name}' to '{modification_name}': "
                            "key already exists in modifications_kwargs."
                        )
                    modifications_kwargs[modification_name] = modifications_kwargs.pop(
                        name
                    )

                if base_name in Alchemist._modification_factories:
                    factory = self._modification_factories[base_name]
                    modification = factory.create_modification(
                        modification_name=modification_name
                    )
                    self.add_modification_to_graph(modification, lambda_value=None)
                else:
                    raise ValueError(
                        f"Modification {base_name} not found in the factories."
                    )

        # After constructing the graph, remove dependencies to skip
        ref_graph = _deepcopy(self._graph)
        for _, data in ref_graph.nodes.data():
            modification = data["modification"]
            if modification.skip_dependencies:
                for skip_dependency in modification.skip_dependencies:
                    self.remove_modification_from_graph(skip_dependency)

        # After constructing the graph, remove redundant modifications
        # Redundancy is determined using a binary overlap principle:
        # - Total overlap: if two modifications have identical alchemical atoms, keep only one
        # - No overlap: if the alchemical atoms are disjoint, keep both modifications
        # - Partial overlap: if the alchemical atoms partially overlap, raise an error
        #   (this behavior may be implemented in the future)
        modifications_kwargs = modifications_kwargs or {}
        redundant_modifications = [
            name for name in self._graph.nodes if name not in lambda_schedule
        ]
        mod_atoms = {
            name: set(modifications_kwargs.get(name, {}).get("alchemical_atoms", []))
            for name in redundant_modifications
        }
        to_remove = set()
        for i, name in enumerate(redundant_modifications):
            base_name = name.split(":", 1)[0]
            set_a = mod_atoms[name]
            for j in range(i + 1, len(redundant_modifications)):
                other_name = redundant_modifications[j]
                other_base_name = other_name.split(":", 1)[0]
                if base_name != other_base_name:
                    continue
                set_b = mod_atoms[other_name]
                if set_a == set_b:
                    # If both modifications have the same alchemical atoms, keep only one
                    to_remove.add(other_name)
                elif not set_a.isdisjoint(set_b):
                    # Partial overlap detected, raise an error
                    raise ValueError(
                        f"Partial overlap detected between modifications '{name}' and '{other_name}'. "
                        "Please ensure that alchemical atoms either fully overlap or are disjoint."
                    )

        for name in to_remove:
            self.remove_modification_from_graph(name)

        for _, data in list(self._graph.nodes.data()):
            modification = data["modification"]

        logger.debug("Created graph of alchemical modifications:\n")
        for line in nx.generate_network_text(
            self._graph, vertical_chains=False, ascii_only=True
        ):
            logger.debug(line)
        logger.debug("")
        return self._graph

    def apply_modifications(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
        modifications_kwargs: Dict[str, Dict[str, Any]],
        *args,
        **kwargs,
    ) -> _mm.System:
        """
        Apply the alchemical modifications to the system.

        Parameters
        ----------
        system : openmm.System
            The system to be modified.
        lambda_schedule : dict
            A dictionary of λ values to be applied to the system.
        modifications_kwargs : dict
            A dictionary of keyword arguments for the modifications.
            It is structured as follows:
            {
                "modification_name": {
                    "key1": value1,
                    "key2": value2,
                    ...
                },
                ...
            }
        args : list
            Additional arguments to be passed to the modifications.
        kwargs : dict
            Additional keyword arguments to be passed to the modifications.

        Returns
        -------
        openmm.System
            The modified system.
        """
        for mod in nx.topological_sort(self._graph):
            lambda_value = self._graph.nodes[mod]["lambda_value"]
            mod_instance = self._graph.nodes[mod]["modification"]
            # Try both instance name and base name for kwargs lookup
            mod_kwargs = modifications_kwargs.get(
                mod, modifications_kwargs.get(mod_instance.NAME, {})
            )

            if lambda_value is None:
                logger.debug(f"Applying {mod} modification")
            else:
                logger.debug(f"Applying {mod} modification with λ={lambda_value}")
            system = mod_instance.apply(
                system,
                alchemical_atoms=alchemical_atoms,
                lambda_value=lambda_value,
                *args,
                **mod_kwargs,
                **kwargs,
            )

        return system


# Register any alchemical modifications defined by entry points.
for modification in entry_points(group="alchemical.modifications"):
    Alchemist.register_modification_factory(modification.name, modification.load()())
