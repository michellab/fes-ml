"""Base strategy for alchemical state creation."""

class AlchemicalStateCreationStrategy:
    def create_alchemical_state(self, *args, **kwargs):
        raise NotImplementedError("create_alchemical_state method must be implemented")