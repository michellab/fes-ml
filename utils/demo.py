import math
import os
import sys

import numpy as np

import openmm
from openmm.app import *

from emle.calculator import EMLECalculator

import sire as sr

# Number of windows for psi and phi.
m = 36
# Total number of windows.
M = m * m

# Equilibrium value for both psi and phi in biasing potentials.
psi = np.linspace(-math.pi, math.pi, m, endpoint=False)
phi = np.linspace(-math.pi, math.pi, m, endpoint=False)

# Get the start and end indices.
psi_index = int(sys.argv[1])
phi_index = int(sys.argv[2])

# Create the output directory.
os.makedirs("./output/traj", exist_ok=True)

# Add harmonic biasing potentials on two dihedrals of dialanine (psi, phi)
# in the OpenMM system for dihedral psi.
bias_torsion_psi = openmm.CustomTorsionForce(
    "0.5*k_psi*dtheta^2; dtheta = min(tmp, 2*pi-tmp); tmp = abs(theta - psi)"
)
bias_torsion_psi.addGlobalParameter("pi", math.pi)
bias_torsion_psi.addGlobalParameter("k_psi", 1.0)
bias_torsion_psi.addGlobalParameter("psi", 0.0)
# 4, 6, 8, 14 are indices of the atoms of the torsion psi.
bias_torsion_psi.addTorsion(4, 6, 8, 14)

# Umbrella force constants for psi and phi.
k_psi = 100
k_phi = 100

# For dihedral phi.
bias_torsion_phi = openmm.CustomTorsionForce(
    "0.5*k_phi*dtheta^2; dtheta = min(tmp, 2*pi-tmp); tmp = abs(theta - phi)"
)
bias_torsion_phi.addGlobalParameter("pi", math.pi)
bias_torsion_phi.addGlobalParameter("k_phi", 1.0)
bias_torsion_phi.addGlobalParameter("phi", 0.0)
# 6, 8, 14, 16 are indices of the atoms of the torsion phi.
bias_torsion_phi.addTorsion(6, 8, 14, 16)

# Load the dialanine system.
mols = sr.load(
    ["ala.top", f"input/ala_{psi_index}_{phi_index}.rst7"],
    show_warnings=False,
    map={"make_whole": True},
)

# Load the topology with OpenMM too.
prm = AmberPrmtopFile("ala.top")

# Create a calculator.
calculator = EMLECalculator(device="cpu")

# Create an EMLEEngine bound to the calculator.
mols, engine = sr.qm.emle(mols, mols[0], calculator)

# Create a QM/MM dynamics object.
d = mols.dynamics(
    timestep="1fs",
    constraint="none",
    cutoff_type="pme",
    integrator="langevin_middle",
    temperature="298.15K",
    qm_engine=engine,
    platform="cpu",
    map={"threads": 1},
)

# Minimise.
d.minimise()

# Get the underlying OpenMM context.
context = d._d._omm_mols

# Get the OpenMM system.
omm_system = context.getSystem()

# Store a copy of the integrator.
integrator = context.getIntegrator().__copy__()

# Add the forces to the OpenMM system.
omm_system.addForce(bias_torsion_psi)
omm_system.addForce(bias_torsion_phi)

# Create a new context.
new_context = openmm.Context(omm_system, integrator, context.getPlatform())

# Set force constant K for the biasing potential.
# The unit here is kJ*mol^{-1}*nm^{-2}, which is the default unit used in OpenMM.
new_context.setParameter("k_psi", k_psi)
new_context.setParameter("k_phi", k_phi)

print(
    f"Sampling at psi index: {psi_index} out of {m}, phi index: {phi_index} out of {m}"
)

# Set the center of the biasing potential.
new_context.setParameter("psi", psi[psi_index])
new_context.setParameter("phi", phi[phi_index])

# Set the postions of the new context to be the same as the original context.
new_context.setPositions(context.getState(getPositions=True).getPositions())

# Sampling production. Trajectories are saved in dcd files.
file_handle = open(f"./output/traj/traj_psi_{psi_index}_phi_{phi_index}.dcd", "bw")
dcd_file = DCDFile(file_handle, prm.topology, dt=integrator.getStepSize())
for x in range(100):
    integrator.step(100)
    state = new_context.getState(getPositions=True)
    positions = state.getPositions()
    dcd_file.writeModel(positions)
file_handle.close()
