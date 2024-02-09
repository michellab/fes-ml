import logging

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from .alchemical_state import AlchemicalState
from .alchemist import Alchemist


class FES:
    def __init__(
        self,
        topology_format=None,
        top_file=None,
        crd_format=None,
        crd_file=None,
        charmm_param_file=None,
        xml_file=None,
        platform_name="Reference",
        system=None,
        integrator=None,
        platform=None,
        context=None,
        topology=None,
        integrator_params={
            "temperature": 300.0 * unit.kelvin,
            "stepSize": 0.001 * unit.picoseconds,
            "frictionCoeff": 2.0 / unit.picoseconds,
        },
        create_system_params={
            "nonbondedMethod": app.PME,
            "nonbondedCutoff": 1.2 * unit.nanometer,
            "constraints": None,
            "rigidWater": True,
        },
    ):
        self.topology_format = topology_format
        self.crd_format = crd_format
        self.xml_file = xml_file
        self.top_file = top_file
        self.crd_file = crd_file
        self.charmm_param_file = charmm_param_file

        # OpenMM essential object instances
        self.system = system
        self.integrator = integrator
        self.platform = platform
        self.context = context
        self.topology = topology

        # Platform-specific variables
        self.platform_name = platform_name if platform_name is not None else "Reference"

        # Params to be passed to OpenMM
        self._create_system_params = create_system_params
        self._integrator_params = integrator_params

        # Batch of alchemical_states
        self._alchemical_states = []

        # Create the OpenMM system
        self._create_system(self._integrator_params, self._create_system_params)

    def _create_system(self, integrator_params=None, create_system_params=None):
        """
        Create an OpenMM system.

        Parameters
        ----------
        integrator_params : dict
            Keyword arguments passed to the simtk.openmm.openmm.LangevinIntegrator
        create_system_params : dict
            Keyword arguments passed to simtk.openmm.app.amberprmtopfile.createSystem

        Returns
        -------
        system : openmm.System
            OpenMM system created.
        """
        from openmm import XmlSerializer

        assert self.topology_format is not None, "No topology format was provided."
        assert self.crd_format is not None, "No coordinate format was provided."

        if self.topology_format.upper() in ["AMBER", "GROMACS", "CHARMM", "PDB"]:
            assert (
                self.top_file is not None
            ), "Topology format is {} but no topology file was provided.".format(
                self.topology_format
            )
        else:
            raise NotImplementedError(
                "Topology format {} is not known.".format(self.topology_format)
            )

        assert (
            self.crd_file is not None
        ), "create_system flag is True but no crd_file was provided."
        if self.platform_name is None:
            logging.info("No platform set. Will use reference.")
            self.platform_name = "Reference"
        else:
            assert self.platform_name in [
                "Reference",
                "CPU",
                "OpenCL",
                "CUDA",
            ], """create_system flag is True but no
               correct platform was provided."""

        # Read topology
        if self.topology_format.upper() == "AMBER":
            if self.topology is None:
                top = app.AmberPrmtopFile(self.top_file)
                self.topology = top.topology
        elif self.topology_format.upper() == "GROMACS":
            if self.topology is None:
                top = app.GromacsTopFile(self.top_file)
                self.topology = top.topology
        elif self.topology_format.upper() == "CHARMM":
            if self.topology is None:
                top = app.CharmmPsfFile(self.top_file)
                self.topology = top.topology
                charmm_params = app.CharmmParameterSet(
                    "charmm.rtf", self.charmm_param_file
                )
        elif self.topology_format.upper() == "PDB":
            if self.topology is None:
                top = app.PDBFile(self.top_file)
                self.topology = top.topology
        else:
            raise NotImplementedError(
                "Topology format {} is not currently supported.".format(
                    self.topology_format
                )
            )

        # Read coordinate file
        if self.crd_format.upper() == "AMBER":
            crd = app.AmberInpcrdFile(self.crd_file)
        elif self.crd_format.upper() == "GROMACS":
            crd = app.GromacsGroFile(self.crd_file)
        elif self.crd_format.upper() == "CHARMM":
            crd = app.CharmmCrdFile(self.crd_file)
        elif self.crd_format.upper() == "PDB":
            crd = app.PDBFile(self.crd_file)
        else:
            raise NotImplementedError(
                "Coordinate format {} is not currently supported.".format(
                    self.crd_format
                )
            )

        if self.system is None:
            if self.xml_file is None:
                assert (
                    create_system_params is not None
                ), "No settings to create the system were provided."

                logging.info(
                    "Creating OpenMM System from {} file.".format(self.topology_format)
                )
                if self.topology_format.upper() == "CHARMM":
                    self.system = top.createSystem(
                        charmm_params, **create_system_params
                    )
                else:
                    self.system = top.createSystem(**create_system_params)
            else:
                logging.info("Creating OpenMM System from XML file.")
                xml_file = open(self.xml_file)
                self.system = XmlSerializer.deserializeSystem(xml_file.read())
                xml_file.close()

        if self.integrator is None:
            assert (
                integrator_params is not None
            ), "No settings to create the integrator were provided."

            self.integrator = mm.LangevinMiddleIntegrator(
                integrator_params["temperature"],
                integrator_params["frictionCoeff"],
                integrator_params["stepSize"],
            )
            logging.info("Creating OpenMM integrator.")
        if self.platform is None:
            self.platform = mm.Platform.getPlatformByName(self.platform_name)
            logging.info("Creating OpenMM platform.")
        if self.context is None:
            self.context = mm.Context(self.system, self.integrator, self.platform)
            logging.info("Creating OpenMM Context.")

        # Set positions in context
        self.context.setPositions(crd.positions)

        return self.system

    def create_alchemical_states(
        self,
        lambda_dict,
        alchemical_atoms,
    ):
        """
        Create a batch of alchemical states for the given lambda values.

        Parameters
        ----------
        lambda_dict : dict
            Dictionary with the lambda values for the alchemical states.
        alchemical_atoms : list
            List of atom indices to be alchemically modified.

        Returns
        -------
        alchemical_states : list of AlchemicalStates
            List of alchemical states.
        """
        lambda_x = lambda_dict["lambda_q"]
        lambda_u = lambda_dict["lambda_lj"]
        lambda_i = lambda_dict["lambda_interpolate"]

        alchemist = Alchemist(self.system, alchemical_atoms)
        for i, x, u in zip(lambda_i, lambda_x, lambda_u):
            lambda_state = {"lambda_q": x, "lambda_lj": u, "lambda_interpolate": i}
            system = alchemist.alchemify(
                i, u, x, ml_potential="ani2x", topology=self.topology
            )

            self._alchemical_states.append(AlchemicalState(lambda_state, system))

            print(f"Created {self._alchemical_states[-1]}")
        return self._alchemical_states

    def create_simulation_batch(self):
        """
        Create a batch of simulations.

        Returns
        ----------
        lambda_dict : dict
            Dictionary with the lambda values for the alchemical states.

        """
        from copy import deepcopy

        for alc in self._alchemical_states:
            simulation = app.Simulation(
                deepcopy(self.topology),
                alc.system,
                deepcopy(self.integrator),
                platform=self.platform,
            )
            alc.simulation = simulation

    def run_equilibration_batch(self, nsteps, minimize=True):
        """
        Run a batch of equilibration simulations.

        Parameters
        ----------
        nsteps : int
            Number of steps to run the simulations.
        minimize : bool, optional, default=True
            If True, the energy will be minimized before running the simulation.

        Returns
        -------
        simulations : list of openmm.app.Simulation
            List of equilibrated simulations.
        """
        if len(self._alchemical_states) == 0:
            raise ValueError("No alchemical states were found.")

        for alc in self._alchemical_states:
            print(f"Equilibrating {alc}")
            sim = alc.simulation

            sim.context.setPositions(
                self.context.getState(getPositions=True).getPositions()
            )

            sim.context.setVelocitiesToTemperature(self.integrator.getTemperature())

            if minimize:
                sim.minimizeEnergy()

            sim.step(nsteps)

        print("Finished all equilibrations!")
        return self._alchemical_states

    def run_simulation_batch(self, niterations, nsteps):
        """
        Run a batch of simulations.

        Parameters
        ----------
        niterations : int
            Number of iterations to run the simulations.

        nsteps : int
            Number of steps to run the simulations.

        Returns
        -------
        u_kln : np.ndarray
            Array with the potential energies of the simulations.
        """
        if len(self._alchemical_states) == 0:
            raise ValueError("No alchemical states were found.")

        nstates = len(self._alchemical_states)
        u_kln = np.zeros([nstates, nstates, niterations], np.float64)
        kT = (
            unit.AVOGADRO_CONSTANT_NA
            * unit.BOLTZMANN_CONSTANT_kB
            * self.integrator.getTemperature()
        )

        U_kn = [[] for k in range(nstates)]
        for k, alc in enumerate(self._alchemical_states):
            sim = alc.simulation

            for iteration in range(niterations):
                print(f"{alc} iteration {iteration} / {niterations}")
                sim.step(nsteps)
                positions = sim.context.getState(getPositions=True).getPositions()
                pbc = sim.context.getState().getPeriodicBoxVectors()

                # Compute energies at all alchemical states
                for l, alc_ in enumerate(self._alchemical_states):
                    sim_ = alc_.simulation
                    sim_.context.setPositions(positions)
                    sim_.context.setPeriodicBoxVectors(*pbc)
                    u_kln[k, l, iteration] = (
                        sim_.context.getState(getEnergy=True).getPotentialEnergy() / kT
                    )

        np.save("u_kln.npy", u_kln)

        return U_kn
