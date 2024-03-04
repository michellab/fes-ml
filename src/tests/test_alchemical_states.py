import os
from typing import Iterable

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
        "platform": "reference",
        "perturbable_constraint": "none",
        "map": {"use_dispersion_correction": True, "tolerance": 0.0005},
    }

    _EMLE_KWARGS = {
        "method": "electrostatic",
    }

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
        lambda_schedule = {"lambda_lj": [1], "lambda_q": [1]}

        # Create the alchemical state
        fes.create_alchemical_states(
            alchemical_atoms=alchemical_atoms,
            lambda_schedule=lambda_schedule,
            dynamics_kwargs=self._DYNAMICS_KWARGS,
            emle_kwargs=self._EMLE_KWARGS,
        )
        fes.run_minimization_batch(max_iterations=10)
        alc = fes.alchemical_states[0]
        alc_energy_decom = {}

        # Energy decomposition
        for i, force in enumerate(alc.system.getForces()):
            force.setForceGroup(i)
        alc.context.reinitialize(preserveState=True)
        for i in range(alc.system.getNumForces()):
            force_name = alc.system.getForce(i).getName()
            energy = alc.context.getState(
                getEnergy=True, groups={i}
            ).getPotentialEnergy()
            alc_energy_decom[force_name] = energy

        # Create a system fully treated at the MM level using OpenMM
        prmtop = _app.AmberPrmtopFile(top_file)
        inpcrd = _app.AmberInpcrdFile(crd_file)
        system = prmtop.createSystem(
            nonbondedMethod=_app.PME,
            nonbondedCutoff=1.2 * _unit.nanometer,
            rigidWater=True,
        )
        integrator = _mm.LangevinMiddleIntegrator(
            298.15 * _unit.kelvin, 1.0 / _unit.picosecond, 1.0 * _unit.femtosecond
        )
        context = _mm.Context(
            system, integrator, _mm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(alc.context.getState(getPositions=True).getPositions())

        # Energy decomposition
        mm_energy_decom = {}
        for i, force in enumerate(system.getForces()):
            force.setForceGroup(i)
        context.reinitialize(preserveState=True)
        for i in range(system.getNumForces()):
            force_name = system.getForce(i).getName()
            energy = context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
            mm_energy_decom[force_name] = energy

        print("----" * 10)
        print("Alchemical energy decomposition")
        for force, energy in alc_energy_decom.items():
            print(force, energy)
        print("----" * 10)
        print("MM energy decomposition")
        for force, energy in mm_energy_decom.items():
            print(force, energy)
        print("----" * 10)

        # Compare the energy components
        bonded_forces = ["HarmonicAngleForce", "PeriodicTorsionForce"]
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
