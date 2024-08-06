from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from openmm import unit
from pymbar import MBAR, timeseries


def main(args):
    temperature = args.temperature
    U_kn = np.load(f"{args.folder}/U_kln_agg.npy")

    # Reformat array such that data is organised in the following away
    # U_kn = [[U^{l}_{k,n} for n in nsamples for k in nstates] for l in nstates]
    # where l (superscript) is the alchemical state at which the potential energy is evaluated
    # and k (subscript) the alchemical state at which it is sampled
    nstates, nstates, nsamples = U_kn.shape
    U_kn = U_kn.transpose(1, 0, 2)
    U_kn = U_kn.reshape(nstates, nstates * nsamples)

    # Keep it in here to contemplate case where number of samples per alchemical state differs
    # N_k = [ U_kn.shape[1]//nstates for _ in range(nstates)]
    N_k = [nsamples for _ in range(nstates)]

    # Compute the overal
    mbar = MBAR(U_kn, N_k, solver_protocol="robust")
    overlap = mbar.compute_overlap()
    plt.figure()
    plt.title("Overlap")
    plt.imshow(overlap["matrix"], vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig("overlap.png")

    # If this fails try setting compute_uncertainty to false
    # See this issue: https://github.com/choderalab/pymbar/issues/419
    results = mbar.compute_free_energy_differences(compute_uncertainty=True)

    # Calculate the free energy
    kT = (
        unit.BOLTZMANN_CONSTANT_kB
        * unit.AVOGADRO_CONSTANT_NA
        * unit.kelvin
        * temperature
    )
    print(
        "Free energy = {}".format(
            (results["Delta_f"][nstates - 1, 0] * kT).in_units_of(
                unit.kilocalorie_per_mole
            )
        )
    )
    print(
        "Statistical uncertainty = {}".format(
            (results["dDelta_f"][nstates - 1, 0] * kT).in_units_of(
                unit.kilocalorie_per_mole
            )
        )
    )


if __name__ == "__main__":
    # accept all options as arguments
    parser = ArgumentParser(description="analyse the aggregated the output")
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        type=str,
        default=None,
        help="folder path for the run. Should have the aggregated output file. ",
    )
    parser.add_argument(
        "-T",
        "--temperature",
        dest="temperature",
        type=float,
        default=298.15,
        help="temperature of the aggregated runs",
    )
    args = parser.parse_args()

    main(args)
