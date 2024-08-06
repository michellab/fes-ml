import glob
from argparse import ArgumentParser

import numpy as np


def main(args):
    folder = args.folder

    U_kln = []
    # number of frames discarded
    frames_disc = 100
    step = 1
    for i in range(100):
        out = f"{folder}/{i}"
        try:
            f = glob.glob(out + "/*npy")[0]
        except:
            break
        U_kln.append(np.load(f)[:, frames_disc::step])

    U_kln = np.asarray(U_kln)
    print(U_kln.shape)
    np.save(f"{folder}/U_kln_agg.npy", U_kln)


if __name__ == "__main__":
    # accept all options as arguments
    parser = ArgumentParser(description="aggregate the output")
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        type=str,
        default=None,
        help="folder path for the run. Should have the files of each window saved as X.npy ",
    )
    args = parser.parse_args()

    main(args)
