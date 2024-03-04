import os
from typing import Any, Dict, Iterable

import numpy as _np
import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit
import pytest

from fes_ml.fes import FES

# Get the directory of the current script
test_data_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(test_data_dir, "test_data")


@pytest.mark.parametrize(
    ("top_file", "crd_file", "alchemical_atoms"),
    [
        (
            os.path.join(test_data_dir, "benzene_gaff2_water.prm7"),
            os.path.join(test_data_dir, "benzene_gaff2_water.rst7"),
            list(range(12)),
        ),
    ],
)
class TestAlchemicalStates:
    _DYNAMICS_KWARGS = {
        "timestep": "1fs",
        "cutoff_type": "PME",
        "cutoff": "12A",
        "constraint": "h_bonds",
        "integrator": "langevin_middle",
        "temperature": "298.15K",
        "platform": "cuda",
        "perturbable_constraint": "none",
        "map": {"use_dispersion_correction": True, "tolerance": 0.0005},
    }

    _EMLE_KWARGS = {
        "method": "electrostatic",
    }

    def _get_energy_decomposition(
        self, system: _mm.System, context: _mm.Context
    ) -> Dict[str, _unit.Quantity]:
        """
        Compute the energy decomposition of a system.

        Parameters
        ----------
        system : openmm.System
            The system to decompose.
        context : openmm.Context
            The context of the system.

        Returns
        -------
        energy_decom : dict
            The energy decomposition of the system.
        """
        energy_decom = {}
        for i, force in enumerate(system.getForces()):
            force.setForceGroup(i)
        context.reinitialize(preserveState=True)
        print("----" * 10)
        for i in range(system.getNumForces()):
            force_name = system.getForce(i).getName()
            energy = context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
            energy_decom[force_name] = energy
            print(force_name, energy)
        print("----" * 10)
        return energy_decom

    def _create_alchemical_states(
        self,
        top_file: str,
        crd_file: str,
        alchemical_atoms: Iterable[int],
        lambda_schedule: Dict[str, Iterable[Any]],
    ):
        """
        Create alchemical states for a given lambda schedule.

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        alchemical_atoms : iterable of int
            The list of alchemical atoms.
        lambda_schedule : dict
            The lambda schedule for the alchemical states.
        """
        # Create the FES object to run the simulations
        fes = FES(top_file=top_file, crd_file=crd_file)

        # Create the alchemical state
        fes.create_alchemical_states(
            alchemical_atoms=alchemical_atoms,
            lambda_schedule=lambda_schedule,
            dynamics_kwargs=self._DYNAMICS_KWARGS,
            emle_kwargs=self._EMLE_KWARGS,
        )
        fes.run_minimization_batch(max_iterations=100)

        return fes

    def _create_openmm_system(self, top_file: str, crd_file: str):
        """
        Create an OpenMM system.

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.

        Returns
        -------
        system : openmm.System
            The OpenMM system.
        """
        prmtop = _app.AmberPrmtopFile(top_file)
        inpcrd = _app.AmberInpcrdFile(crd_file)
        system = prmtop.createSystem(
            nonbondedMethod=_app.PME,
            nonbondedCutoff=1.2 * _unit.nanometer,
            rigidWater=True,
        )
        context = _mm.Context(
            system,
            _mm.LangevinMiddleIntegrator(
                298.15 * _unit.kelvin, 1.0 / _unit.picosecond, 1.0 * _unit.femtosecond
            ),
            _mm.Platform.getPlatformByName("CUDA"),
        )
        return system, context

    def test_alchemical_mm(
        self, top_file: str, crd_file: str, alchemical_atoms: Iterable[int]
    ):
        """
        Test that a state where LJ and charges are alchemified is created correctly.

        Notes
        -----
        This test compares the energy components of a system created where the LJ (lambda_lj=1)
        and charges (lambda_q=1) are fully turned to the energy components of a system fully
        treated at the MM level.

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        alchemical_atoms : iterable of int
            The list of alchemical atoms.
        """
        # Create the FES object to run the simulations
        fes = FES(top_file=top_file, crd_file=crd_file)

        # Create a system where LJ and charges are fully turned on
        lambda_schedule: Dict[str, Iterable[Any]] = {"lambda_lj": [1], "lambda_q": [1]}

        # Create the alchemical state
        fes = self._create_alchemical_states(
            top_file=top_file,
            crd_file=crd_file,
            alchemical_atoms=alchemical_atoms,
            lambda_schedule=lambda_schedule,
        )
        alc = fes.alchemical_states[0]

        # Energy decomposition
        alc_energy_decom = self._get_energy_decomposition(alc.system, alc.context)

        # Create a system fully treated at the MM level using OpenMM
        system, context = self._create_openmm_system(top_file, crd_file)
        context.setPositions(alc.context.getState(getPositions=True).getPositions())

        # Energy decomposition
        mm_energy_decom = self._get_energy_decomposition(system, context)

        # Compare the energy components
        bonded_forces = [
            "HarmonicBondForce",
            "HarmonicAngleForce",
            "PeriodicTorsionForce",
        ]
        for force in bonded_forces:
            assert _np.isclose(
                alc_energy_decom[force]._value, mm_energy_decom[force]._value
            ), f"Energy of {force} is not the same."

        non_bonded_forces = [
            "NonbondedForce",
            "CustomBondForce",
            "CustomNonbondedForce",
        ]
        alc_nonbonded_force = sum(
            alc_energy_decom[force]._value
            for force in alc_energy_decom
            if force in non_bonded_forces
        )
        assert _np.isclose(
            alc_nonbonded_force, mm_energy_decom["NonbondedForce"]._value
        ), "Nonbonded energy is not the same."

    def test_alchemical_ml(
        self, top_file: str, crd_file: str, alchemical_atoms: Iterable[int]
    ):
        pass
