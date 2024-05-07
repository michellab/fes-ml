"""Sire alchemical state creation strategy."""

import logging
from copy import deepcopy as _deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as _np
import openmm as _mm
import sire as _sr
from emle.calculator import EMLECalculator as _EMLECalculator

from ...utils import energy_decomposition as energy_decomposition
from ..alchemical_state import AlchemicalState
from ..alchemist import Alchemist
from .base_strategy import AlchemicalStateCreationStrategy

logger = logging.getLogger(__name__)


class SireCreationStrategy(AlchemicalStateCreationStrategy):
    """Strategy for creating alchemical states using Sire."""

    def create_alchemical_state(
        self,
        top_file: str,
        crd_file: str,
        alchemical_atoms: List[int],
        lambda_schedule: Dict[str, Union[float, int]],
        minimise_iterations: int = 1,
        ml_potential: str = "ani2x",
        ml_potential_kwargs: Optional[Dict[str, Any]] = None,
        create_system_kwargs: Optional[Dict[str, Any]] = None,
        topology: Optional[_mm.app.topology.Topology] = None,
        dynamics_kwargs: Optional[Dict[str, Any]] = None,
        emle_kwargs: Optional[Dict[str, Any]] = None,
        integrator: Optional[Any] = None,
    ) -> AlchemicalState:
        """
        Create an alchemical state for the given lambda values using OpenMM Systems created with Sire.

        Parameters
        ----------
        top_file : str
            Path to the topology file.
        crd_file : str
            Path to the coordinate file.
        alchemical_atoms : list of int
            A list of atom indices to be alchemically modified.
        lambda_schedule : dict
            A dictionary mapping the name of the alchemical modification to the lambda value.
        minimise_iterations : int, optional, default=1
            The number of minimisation iterations to perform before creating the alchemical state.
            1 step is enough to bring the geometry to the distances imposed by the restraints.
            If None, no minimisation is performed.
        ml_potential : str, optional, default='ani2x'
            The machine learning potential to use in the mechanical embedding scheme.
        ml_potential_kwargs : dict, optional, default=None
            Additional keyword arguments to be passed to MLPotential when creating the ML potential in OpenMM-ML.
            See: https://openmm.github.io/openmm-ml/dev/generated/openmmml.MLPotential.html
        create_system_kwargs : dict, optional, default=None
            Additional keyword arguments to be passed to the createSystem or createMixedSystem methods of the
            OpenMM-ML package. See: https://openmm.github.io/openmm-ml/dev/generated/openmmml.models.macepotential.MACEPotentialImpl.html
        topology : openmm.app.Topology, optional, default=None
            The OpenMM topology object.
        dynamics_kwargs : dict
            Additional keyword arguments to be passed to sire.mol.Dynamics.
            See https://sire.openbiosim.org/api/mol.html#sire.mol.Dynamics.
        emle_kwargs : dict
            Additional keyword arguments to be passed to the EMLECalculator.
            See TODO.
        integrator : Any, optional, default=None
            The OpenMM integrator to use. If None, the integrator is the one used in the dynamics_kwargs, if provided.
            Otherwise, the default is a LangevinMiddle integrator with a 1 fs timestep and a 298.15 K temperature.

        Returns
        -------
        AlchemicalState
            The alchemical state.
        """

        if dynamics_kwargs is None:
            dynamics_kwargs = {
                "timestep": "1fs",
                "constraint": "none",
                "cutoff_type": "pme",
                "cutoff": "9A",
                "integrator": "langevin_middle",
                "temperature": "298.15K",
            }
        else:
            dynamics_kwargs = _deepcopy(dynamics_kwargs)

        if emle_kwargs is None:
            emle_kwargs = {"method": "electrostatic", "backend": "torchani"}
        else:
            emle_kwargs = _deepcopy(emle_kwargs)

        logger.debug("-" * 100)
        logger.debug("Creating alchemical state using SireCreationStrategy.")
        logger.debug(f"top_file: {top_file}")
        logger.debug(f"crd_file: {crd_file}")
        logger.debug(f"alchemical_atoms: {alchemical_atoms}")
        logger.debug(f"lambda_schedule: {lambda_schedule}")
        logger.debug(f"ml_potential: {ml_potential}")
        logger.debug(f"topology: {topology}")
        logger.debug("dynamics_kwargs:")
        for key, value in dynamics_kwargs.items():
            logger.debug(f"{key}: {value}")
        logger.debug("emle_kwargs:")
        for key, value in emle_kwargs.items():
            logger.debug(f"{key}: {value}")

        # Load the molecular system.
        mols = _sr.load(top_file, crd_file, show_warnings=True)

        # Create a QM/MM dynamics object
        d = mols.dynamics(**dynamics_kwargs)

        if minimise_iterations:
            d.minimise(minimise_iterations)

        # Get the underlying OpenMM context.
        omm = d._d._omm_mols

        # Get the OpenMM system.
        system = omm._system

        # Create an Alchemist object with the modifications to apply
        alchemist = Alchemist()
        alchemist.create_alchemical_graph(lambda_schedule=lambda_schedule)
        alchemist.plot_graph()
        alchemist.apply_modifications(
            system=system,
            alchemical_atoms=alchemical_atoms,
            topology=topology,
            ml_potential=ml_potential,
        )

        if integrator is None:
            # Create a new integrator
            integrator = omm.getIntegrator().__copy__()
        else:
            integrator = _deepcopy(integrator)

        # Create a new context and set positions and velocities
        context = _mm.Context(system, integrator, omm.getPlatform())
        context.setPositions(omm.getState(getPositions=True).getPositions())

        try:
            context.setVelocitiesToTemperature(integrator.getTemperature())
        except AttributeError:
            context.setVelocitiesToTemperature(
                float(dynamics_kwargs["temperature"][:-1])
            )

        logger.debug("Energy decomposition of the system:")
        logger.debug(
            f"Total potential energy: {context.getState(getEnergy=True).getPotentialEnergy()}"
        )
        energy_decomp = energy_decomposition(system, context)
        for force, energy in energy_decomp.items():
            logger.debug(f"{force}: {energy}")
        logger.debug("Alchemical state created successfully.")
        logger.debug("-" * 100)

        # Create the AlchemicalState
        alc_state = AlchemicalState(
            system=system,
            context=context,
            integrator=integrator,
        )

        return alc_state
