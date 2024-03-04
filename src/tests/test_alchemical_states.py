import os
from typing import Any, Dict, Iterable, Optional

import numpy as _np
import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit
import pytest
from openmmml import MLPotential

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

    _R_TOL = 1e-5

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
        **kwargs,
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
            **kwargs,
        )
        fes.run_minimization_batch(max_iterations=100)

        return fes

    def _create_openmm_system(
        self, top_file: str, crd_file: str, ml_atoms: Optional[Iterable[int]] = None, lambda_interpolate: Optional[float] = None
    ):
        """
        Create an OpenMM system.

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        ml_atoms : iterable of int, optional
            The list of atoms to be treated with the ML potential.
            If None, the system is fully treated at the MM level.
        lambda_interpolate : float, optional
            The value of the lambda interpolation parameter.

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

        if ml_atoms is not None:
            potential = MLPotential("ani2x")
            ml_system = potential.createMixedSystem(
                prmtop.topology, system, ml_atoms, interpolate=1
            )
            system = ml_system

        context = _mm.Context(
            system,
            _mm.LangevinMiddleIntegrator(
                298.15 * _unit.kelvin, 1.0 / _unit.picosecond, 1.0 * _unit.femtosecond
            ),
            _mm.Platform.getPlatformByName("CUDA"),
        )

        if ml_atoms is None and lambda_interpolate is not None:
            context.setParameter("lambda_interpolate", lambda_interpolate)

        return system, context

    def _test_energy_decomposition(
        self,
        top_file: str,
        crd_file: str,
        lambda_schedule: Dict[str, Iterable[Any]],
        alchemical_atoms: Iterable[int],
        ml_atoms: Optional[Iterable[int]] = None,
    ):
        """
        Test the energy decomposition of a system.

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        lambda_schedule : dict
            The lambda schedule for the alchemical states.
        alchemical_atoms : iterable of int
            The list of alchemical atoms.
        ml_atoms : iterable of int, optional
            The list of atoms to be treated with the ML potential.
            If None, the system is fully treated at the MM level.
        """
        # Create the alchemical state
        fes = self._create_alchemical_states(
            top_file=top_file,
            crd_file=crd_file,
            alchemical_atoms=alchemical_atoms,
            lambda_schedule=lambda_schedule,
            topology=_app.AmberPrmtopFile(top_file).topology,
        )
        alc = fes.alchemical_states[0]

        # Energy decomposition of the alchemical state
        alc_energy_decom = self._get_energy_decomposition(alc.system, alc.context)

        # Create a system fully treated at the MM level using OpenMM
        system, context = self._create_openmm_system(
            top_file, crd_file, ml_atoms=ml_atoms
        )
        context.setPositions(alc.context.getState(getPositions=True).getPositions())

        # Energy decomposition of the ML/MM system
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

    def test_alchemical_lj_charges(
        self, top_file: str, crd_file: str, alchemical_atoms: Iterable[int]
    ):
        """
        Test that a system with alchemified LJ, and charges is created correctly.

        Notes
        -----
        This test compares the energy components of a system created with LJ (lambda_lj=1),
        and charges (lambda_q=1) fully turned on, to the energy components of a system fully 
        treated at the ML/MM level

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        alchemical_atoms : iterable of int
            The list of alchemical atoms.
        """
        # Create a system where LJ and charges are fully turned on
        lambda_schedule: Dict[str, Iterable[Any]] = {"lambda_lj": [1], "lambda_q": [1]}
        ml_atoms = None
        self._test_energy_decomposition(
            top_file, crd_file, lambda_schedule, alchemical_atoms, ml_atoms
        )

    def test_alchemical_ml(
        self, top_file: str, crd_file: str, alchemical_atoms: Iterable[int]
    ):
        """
        Test that a system with alchemified MLP is created correctly.

        Notes
        -----
        This test compares the energy components of a system created with the MLP (lambda_interpolate=1)
        fully turned on, to the energy components of a system fully treated at the ML/MM level

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        alchemical_atoms : iterable of int
            The list of alchemical atoms.
        """
        # Create a system where LJ and charges are fully turned on
        lambda_schedule: Dict[str, Iterable[Any]] = {"lambda_interpolate": [1]}
        ml_atoms = alchemical_atoms
        self._test_energy_decomposition(
            top_file, crd_file, lambda_schedule, alchemical_atoms, ml_atoms
        )

    def test_alchemical_ml_lj_charges(
        self, top_file: str, crd_file: str, alchemical_atoms: Iterable[int]
    ):
        """
        Test that a system with alchemified LJ, charges, and the MLP is created correctly.

        Notes
        -----
        This test compares the energy components of a system created with LJ (lambda_lj=1),
        charges (lambda_q=1), and MLP (lambda_interpolate=1) fully turned on, to the energy
        components of a system fully treated at the ML/MM level

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        alchemical_atoms : iterable of int
            The list of alchemical atoms.
        """
        # Create a system where LJ, charges, and the MLP are fully turned on
        lambda_schedule: Dict[str, Iterable[Any]] = {
            "lambda_interpolate": [1],
            "lambda_lj": [1],
            "lambda_q": [1],
        }

        ml_atoms = alchemical_atoms
        self._test_energy_decomposition(
            top_file, crd_file, lambda_schedule, alchemical_atoms, ml_atoms, lambda_interpolate=lambda_schedule["lambda_interpolate"]
        )

        # Create a system where LJ and charges are fully turned on but the MLP is turned off
        # and compare it to the energy of the system fully treated at the ML/MM level
        lambda_schedule: Dict[str, Iterable[Any]] = {
            "lambda_interpolate": [0],
            "lambda_lj": [1],
            "lambda_q": [1],
        }

        ml_atoms = alchemical_atoms
        self._test_energy_decomposition(
            top_file, crd_file, lambda_schedule, alchemical_atoms, ml_atoms, lambda_interpolate=lambda_schedule["lambda_interpolate"]
        )

        # Create a system where LJ and charges are fully turned on but the MLP is turned off
        # and compare it to the energy of the system fully treated at the MM level
        lambda_schedule: Dict[str, Iterable[Any]] = {
            "lambda_interpolate": [0],
            "lambda_lj": [1],
            "lambda_q": [1],
        }

        ml_atoms = alchemical_atoms
        self._test_energy_decomposition(
            top_file, crd_file, lambda_schedule, alchemical_atoms
        )