import os
from typing import Dict, List, Optional, Tuple

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
        "platform": "Reference",
        "perturbable_constraint": "none",
        "map": {"use_dispersion_correction": True, "tolerance": 0.0005},
    }

    _EMLE_KWARGS = {
        "method": "electrostatic",
    }

    _R_TOL = 1e-6
    _A_TOL = (
        1e-4  # used to determine what small values should be considered close to zero
    )

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
        print("------" * 10)
        for i in range(system.getNumForces()):
            force_name = system.getForce(i).getName()
            energy = context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
            energy_decom[force_name] = energy
            print(force_name, energy)
        print("------" * 10)
        return energy_decom

    def _create_alchemical_states(
        self,
        top_file: str,
        crd_file: str,
        alchemical_atoms: List[int],
        lambda_schedule: Dict[str, List[Optional[float]]],
        **kwargs,
    ) -> FES:
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
        fes = FES(
            top_file=top_file,
            crd_file=crd_file,
        )

        # Create the alchemical state
        fes.create_alchemical_states(
            alchemical_atoms=alchemical_atoms,
            lambda_schedule=lambda_schedule,
            dynamics_kwargs=self._DYNAMICS_KWARGS,
            emle_kwargs=self._EMLE_KWARGS,
            **kwargs,
        )

        return fes

    def _create_openmm_system(
        self,
        top_file: str,
        crd_file: str,
        ml_atoms: Optional[List[int]] = None,
        MLInterpolation: Optional[float] = None,
    ) -> Tuple[_mm.System, _mm.Context]:
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
        MLInterpolation : float, optional
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
            _mm.Platform.getPlatformByName("Reference"),
        )

        if ml_atoms is None and MLInterpolation is not None:
            context.setParameter("MLInterpolation", MLInterpolation)

        return system, context

    def _test_energy_decomposition(
        self,
        top_file: str,
        crd_file: str,
        lambda_schedule: Dict[str, List[Optional[float]]],
        alchemical_atoms: List[int],
        ml_atoms: Optional[List[int]] = None,
        MLInterpolation: Optional[float] = None,
        **kwargs,
    ) -> None:
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
        MLInterpolation : float, optional
            The value of the lambda interpolation parameter.
        """
        # Create the alchemical state
        fes = self._create_alchemical_states(
            top_file=top_file,
            crd_file=crd_file,
            alchemical_atoms=alchemical_atoms,
            lambda_schedule=lambda_schedule,
            **kwargs,
        )
        alc = fes.alchemical_states[0]

        # Energy decomposition of the alchemical state
        alc_energy_decom = self._get_energy_decomposition(alc.system, alc.context)

        # Create a system fully treated at the MM level using OpenMM
        system, context = self._create_openmm_system(
            top_file, crd_file, ml_atoms=ml_atoms, MLInterpolation=MLInterpolation
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
                alc_energy_decom[force]._value,
                mm_energy_decom[force]._value,
                atol=self._A_TOL,
                rtol=self._R_TOL,
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
            alc_nonbonded_force,
            mm_energy_decom["NonbondedForce"]._value,
            atol=self._A_TOL,
            rtol=self._R_TOL,
        ), "Nonbonded energy is not the same."

    def test_alchemical_lj_charges(
        self, top_file: str, crd_file: str, alchemical_atoms: List[int]
    ) -> None:
        """
        Test that a system with alchemified LJ, and/or charges is created correctly.

        Notes
        -----
        This test compares the energy components of a system created with LJ (LJSoftCore=1),
        and/or charges (ChargeScaling=1) fully turned on, to the energy components of a system fully
        treated at the MM level

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        alchemical_atoms : iterable of int
            The list of alchemical atoms.
        """
        print("Testing function: test_alchemical_lj_charges")
        # Create a system where LJ and charges are alchemified and fully turned on
        lambda_schedule_lj_q: Dict[str, List[Optional[float]]] = {
            "LJSoftCore": [1],
            "ChargeScaling": [1],
        }
        ml_atoms = None
        self._test_energy_decomposition(
            top_file, crd_file, lambda_schedule_lj_q, alchemical_atoms, ml_atoms
        )

        # Create a system where LJ is alchemified and fully turned on
        lambda_schedule_lj: Dict[str, List[Optional[float]]] = {"LJSoftCore": [1]}
        ml_atoms = None
        self._test_energy_decomposition(
            top_file, crd_file, lambda_schedule_lj, alchemical_atoms, ml_atoms
        )

        # Create a system where charges are alchemified and fully turned on
        lambda_schedule_q: Dict[str, List[Optional[float]]] = {"ChargeScaling": [1]}
        ml_atoms = None
        self._test_energy_decomposition(
            top_file, crd_file, lambda_schedule_q, alchemical_atoms, ml_atoms
        )

    # TODO: make this faster so that CI doesn't take too long
    def test_alchemical_ml(
        self, top_file: str, crd_file: str, alchemical_atoms: List[int]
    ) -> None:
        """
        Test that a system with alchemified MLP is created correctly.

        Notes
        -----
        This test compares the energy components of a system created with the MLP (MLInterpolation=1)
        (MLInterpolation=0) fully turned on (off), to the energy components of a system fully treated
        at the ML/MM (MM) level.

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        alchemical_atoms : iterable of int
            The list of alchemical atoms.
        """
        print("Testing function: test_alchemical_ml")
        # Create a system where the MLP is turned on and compare it to the energy of
        # the system fully treated at the MM/ML level
        lambda_schedule_intp_1: Dict[str, List[Optional[float]]] = {
            "MLInterpolation": [1]
        }
        ml_atoms = alchemical_atoms
        self._test_energy_decomposition(
            top_file,
            crd_file,
            lambda_schedule_intp_1,
            alchemical_atoms,
            ml_atoms,
            ml_potential="ani2x",
        )
    '''
        # Create a system where the MLP is turned off and compare it to the energy of
        # the system fully treated at the MM level
        lambda_schedule_intp_0: Dict[str, List[Optional[float]]] = {
            "MLInterpolation": [0]
        }
        ml_atoms = alchemical_atoms
        self._test_energy_decomposition(
            top_file,
            crd_file,
            lambda_schedule_intp_0,
            alchemical_atoms,
            ml_atoms,
            MLInterpolation=lambda_schedule_intp_0["MLInterpolation"][0],
            ml_potential="ani2x",
        )

    
    def test_alchemical_ml_lj_charges(
        self, top_file: str, crd_file: str, alchemical_atoms: List[int]
    ) -> None:
        """
        Test that a system with alchemified LJ, charges, and the MLP is created correctly.

        Notes
        -----
        This test compares the energy components of a system created with LJ (LJSoftCore=1),
        charges (ChargeScaling=1), and/or MLP (MLInterpolation=1) (MLInterpolation=0) fully
        turned on/off, to the energy components of a system fully treated at the ML/MM (MM) level.

        Parameters
        ----------
        top_file : str
            The topology file of the system.
        crd_file : str
            The coordinate file of the system.
        alchemical_atoms : iterable of int
            The list of alchemical atoms.
        """
        print("Testing function: test_alchemical_ml_lj_charges")
        # Create a system where LJ, charges, and the MLP are fully turned on
        lambda_schedule_intp_1: Dict[str, List[Optional[float]]] = {
            "MLInterpolation": [1],
            "LJSoftCore": [1],
            "ChargeScaling": [1],
        }

        ml_atoms = alchemical_atoms
        self._test_energy_decomposition(
            top_file,
            crd_file,
            lambda_schedule_intp_1,
            alchemical_atoms,
            ml_atoms,
            MLInterpolation=lambda_schedule_intp_1["MLInterpolation"][0],
        )

        # Create a system where LJ and charges are fully turned on but the MLP is turned off
        # and compare it to the energy of the system fully treated at the ML/MM level
        lambda_schedule_intp_0: Dict[str, List[Optional[float]]] = {
            "MLInterpolation": [0],
            "LJSoftCore": [1],
            "ChargeScaling": [1],
        }

        ml_atoms = alchemical_atoms
        self._test_energy_decomposition(
            top_file,
            crd_file,
            lambda_schedule_intp_0,
            alchemical_atoms,
            ml_atoms,
            MLInterpolation=lambda_schedule_intp_0["MLInterpolation"][0],
        )
    '''
