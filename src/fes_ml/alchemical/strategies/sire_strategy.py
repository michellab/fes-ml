"""Sire alchemical state creation strategy."""

import logging
import shutil as _shutil
from copy import deepcopy as _deepcopy
from typing import Any, Dict, List, Optional, Union

import openmm as _mm
import openmm.app as _app
import sire as _sr

from ..alchemical_state import AlchemicalState
from .base_strategy import AlchemicalStateCreationStrategy

logger = logging.getLogger(__name__)


class SireCreationStrategy(AlchemicalStateCreationStrategy):
    """Strategy for creating alchemical states using Sire."""

    _TMP_DIR = "tmp_fes_ml_sire"

    _DYNAMIC_KWARGS = {
        "timestep": "1fs",
        "constraint": "none",
        "cutoff_type": "pme",
        "cutoff": "9A",
        "integrator": "langevin_middle",
        "temperature": "298.15K",
        "perturbable_constraint": None,
    }

    _EMLE_KWARGS = {"method": "electrostatic", "backend": "torchani"}

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
        dynamics_kwargs: Optional[Dict[str, Any]] = None,
        emle_kwargs: Optional[Dict[str, Any]] = None,
        integrator: Optional[Any] = None,
        keep_tmp_files: bool = True,
        *args,
        **kwargs,
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
        dynamics_kwargs : dict
            Additional keyword arguments to be passed to sire.mol.Dynamics.
            See https://sire.openbiosim.org/api/mol.html#sire.mol.Dynamics.
        emle_kwargs : dict
            Additional keyword arguments to be passed to the EMLECalculator.
            See https://github.com/chemle/emle-engine/blob/main/emle/calculator.py#L399-L519.
        integrator : Any, optional, default=None
            The OpenMM integrator to use. If None, the integrator is the one used in the dynamics_kwargs, if provided.
            Otherwise, the default is a LangevinMiddle integrator with a 1 fs timestep and a 298.15 K temperature.
        keep_tmp_files : bool, optional, default=True
            Whether to keep the temporary files created by the strategy.
        args : list
            Additional arguments to be passed to the Alchemist `apply_modifications` method.
        kwargs : dict
            Additional keyword arguments to be passed to the Alchemist `apply_modifications` method.

        Returns
        -------
        AlchemicalState
            The alchemical state.
        """
        logger.debug("=" * 100)
        # Generate local copies of the dynamics and EMLE kwargs
        dynamics_kwargs = (
            _deepcopy(self._DYNAMIC_KWARGS)
            if dynamics_kwargs is None
            else _deepcopy(dynamics_kwargs)
        )
        emle_kwargs = (
            _deepcopy(self._EMLE_KWARGS)
            if emle_kwargs is None
            else _deepcopy(emle_kwargs)
        )
        passed_args = locals()
        passed_args["dynamics_kwargs"] = dynamics_kwargs
        passed_args["emle_kwargs"] = emle_kwargs

        # Report the creation settings
        self._report_creation_settings(passed_args)

        # Load the molecular system.
        mols = _sr.load(top_file, crd_file, show_warnings=True)

        # Select the alchemical subsystem
        alchemical_subsystem = mols.atoms(alchemical_atoms)

        # Write the alchemical subsystem and full system parm7 to temp files
        alchemical_prm7 = _sr.save(
            alchemical_subsystem,
            directory=self._TMP_DIR,
            filename="alchemical_subsystem.prm7",
            format=["prm7"],
        )

        parm7 = _sr.save(
            mols,
            directory=self._TMP_DIR,
            filename="full_system.prm7",
            format=["prm7"],
        )

        # Load the topology of the full system
        topology = _app.AmberPrmtopFile(parm7[0]).topology

        # Create a QM/MM dynamics object
        d = mols.dynamics(**dynamics_kwargs)

        if minimise_iterations:
            d.minimise(minimise_iterations)

        # Get the OpenMM context, system, integrator, and platform
        omm_context = _sr.convert.to(mols, "openmm")
        omm_system = omm_context.getSystem()
        omm_integrator = omm_context.getIntegrator()
        omm_platform = omm_context.getPlatform()

        # Report the energy decomposition before applying the alchemical modifications
        self._report_energy_decomposition(omm_context, omm_system)

        # Run the Alchemist
        self._run_alchemist(
            omm_system,
            alchemical_atoms,
            lambda_schedule,
            *args,
            **kwargs,
        )

        # Create the final integrator
        if integrator is None:
            integrator = _deepcopy(omm_integrator)
        else:
            integrator = _deepcopy(integrator)

        # Create a simulation object
        simulation = _mm.app.Simulation(topology, omm_system, integrator, omm_platform)

        # Set the positions and velocities
        simulation.context.setPositions(
            omm_context.getState(getPositions=True).getPositions()
        )
        try:
            simulation.context.setVelocitiesToTemperature(integrator.getTemperature())
        except AttributeError:
            simulation.context.setVelocitiesToTemperature(
                float(dynamics_kwargs["temperature"][:-1])
            )

        # Report the energy decomposition after applying the alchemical modifications
        self._report_energy_decomposition(simulation.context, omm_system)

        # Create the AlchemicalState
        alc_state = AlchemicalState(
            system=omm_system,
            context=simulation.context,
            integrator=integrator,
            simulation=simulation,
            topology=topology,
            modifications=lambda_schedule,
        )

        if not keep_tmp_files:
            # Clean up the temporary directory
            _shutil.rmtree(self._TMP_DIR)

        logger.debug("Alchemical state created successfully.")
        logger.debug("=" * 100)

        return alc_state
