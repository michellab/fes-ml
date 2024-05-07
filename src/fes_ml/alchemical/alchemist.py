"""Module for the Alchemist class."""
import logging
import sys
from typing import Dict, List, Optional

import networkx as nx
import openmm as _mm

from .modifications.base_modification import BaseModification, BaseModificationFactory

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

logger = logging.getLogger(__name__)


class Alchemist:
    _modification_factories: Dict[str, BaseModificationFactory] = {}

    @staticmethod
    def register_modification_factory(
        name: str, factory: BaseModificationFactory
    ) -> Dict[str, BaseModificationFactory]:
        Alchemist._modification_factories[name] = factory

        return Alchemist._modification_factories

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    def __repr__(self) -> str:
        return nx.to_dict_of_lists(self._graph)

    @property
    def graph(self) -> nx.DiGraph:
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
        self._graph.add_node(
            modification.NAME, modification=modification, lambda_value=lambda_value
        )
        for pre_dependency in modification.pre_dependencies:
            factory = self._modification_factories[pre_dependency]
            pre_modification = factory.create_modification()
            self._graph.add_edge(pre_modification.NAME, modification.NAME)
            self.add_modification_to_graph(pre_modification, None)

        for post_dependency in modification.post_dependencies:
            factory = self._modification_factories[post_dependency]
            post_modification = factory.create_modification()
            self._graph.add_edge(modification.NAME, post_modification.NAME)
            self.add_modification_to_graph(post_modification, None)

    def create_alchemical_graph(
        self,
        lambda_schedule: Dict[str, float],
        additional_modifications: Optional[List[str]] = None,
    ):
        """
        Create a graph of alchemical modifications to apply.

        Parameters
        ----------
        lambda_schedule : dict
            A dictionary of lambda values to be applied to the system.
        additional_modifications : list of str
            Additional modifications to apply.

        Returns
        -------
        nx.DiGraph
            The graph of the modifications to apply.
        """
        for name, lambda_value in lambda_schedule.items():
            if name in Alchemist._modification_factories:
                factory = self._modification_factories[name]
                modification = factory.create_modification()
                self.add_modification_to_graph(modification, lambda_value=lambda_value)
            else:
                raise ValueError(f"Modification {name} not found in the factories.")

        if additional_modifications is not None:
            for name in additional_modifications:
                if name in Alchemist._modification_factories:
                    factory = self._modification_factories[name]
                    modification = factory.create_modification()
                    self.add_modification_to_graph(
                        modification, lambda_value=lambda_value
                    )
                else:
                    raise ValueError(f"Modification {name} not found in the factories.")

        return self._graph

    def apply_modifications(
        self,
        system: _mm.System,
        alchemical_atoms: List[int],
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
        for mod in nx.topological_sort(self._graph):
            lambda_value = self._graph.nodes[mod]["lambda_value"]
            modification = self._graph.nodes[mod]["modification"]
            if lambda_value is None:
                logger.debug(f"Applying {mod} modification")
            else:
                logger.debug(
                    f"Applying {mod} modification with lambda value {lambda_value}"
                )
            system = modification.apply(
                system,
                alchemical_atoms=alchemical_atoms,
                lambda_value=lambda_value,
                *args,
                **kwargs,
            )

        return system


# Register any alchemical modifications defined by entry points.
for modification in entry_points(group="alchemical.modifications"):
    Alchemist.register_modification_factory(modification.name, modification.load()())
