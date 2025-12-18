"""Module that defines the AlchemicalStateFactory class."""

import logging

from .alchemical_state import AlchemicalState
from .strategies import (
    AlchemicalStateCreationStrategy,
    OpenFFCreationStrategy,
    SireCreationStrategy,
)

logger = logging.getLogger(__name__)


class AlchemicalStateFactory:
    """
    Factory to create alchemical states using different strategies.

    Attributes
    ----------
    strategies : dict
        A dictionary with the available strategies to create alchemical states.
        Currently, the available strategies are: "sire".
    """

    def __init__(self) -> None:
        """Initialize the AlchemicalStateFactory."""
        self.strategies: dict = {}

    def register_strategy(self, name: str, strategy: AlchemicalStateCreationStrategy) -> None:
        """
        Register a new strategy to create alchemical states.

        Parameters
        ----------
        name : str
            The name of the strategy.
        strategy : AlchemicalStateCreationStrategy
            The strategy to create alchemical states.
        """
        self.strategies[name] = strategy

    def get_strategy(self, name: str) -> AlchemicalStateCreationStrategy:
        """
        Get a strategy to create alchemical states.

        Parameters
        ----------
        name : str
            The name of the strategy.

        Returns
        -------
        AlchemicalStateCreationStrategy
            The strategy to create alchemical states.
        """
        return self.strategies.get(name, None)

    def create_alchemical_state(self, strategy_name: str = "sire", *args, **kwargs) -> AlchemicalState:
        """
        Create an alchemical state for the given λ values.

        Parameters
        ----------
        strategy_name : str, default='sire'
            The name of the strategy to create the alchemical state.

        Returns
        -------
        AlchemicalState
            The alchemical state.
        """
        strategy = self.get_strategy(strategy_name)
        if strategy:
            alchemical_state = strategy.create_alchemical_state(*args, **kwargs)
            return alchemical_state
        else:
            raise ValueError(f"No strategy found with name {strategy_name}. Available strategies are {list(self.strategies.keys())}.")


# Create the alchemical factory and register the available strategies
alchemical_factory = AlchemicalStateFactory()
alchemical_factory.register_strategy("sire", SireCreationStrategy())
alchemical_factory.register_strategy("openff", OpenFFCreationStrategy())
