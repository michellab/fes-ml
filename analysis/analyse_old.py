# Estimate free energy of Lennard-Jones particle insertion
import numpy as np
from pymbar import MBAR, timeseries

u_kln = np.load("u_kln.npy")
nstates = u_kln.shape[0]

## Subsample data to extract uncorrelated equilibrium timeseries
N_k = np.zeros([nstates], np.int32)  # number of uncorrelated samples
for k in range(nstates):
    [nequil, g, Neff_max] = timeseries.detectEquilibration(u_kln[k, k, :])
    indices = timeseries.subsampleCorrelatedData(u_kln[k, k, :], g=g)
    N_k[k] = len(indices)
    u_kln[k, :, 0 : N_k[k]] = u_kln[k, :, indices].T

# Compute free energy differences
mbar = MBAR(u_kln, N_k)

# dont compute uncertainties here, if you do it may fail with an error for
# pymbar versions > 3.0.3. See this issue: https://github.com/choderalab/pymbar/issues/419
[DeltaF_ij] = mbar.getFreeEnergyDifferences(compute_uncertainty=False)

print("Free energy change to insert a particle = ", DeltaF_ij[nstates - 1][0])
print("DeltaF_ij", DeltaF_ij)
