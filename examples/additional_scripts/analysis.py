import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import rdf as RDF
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.topology.guessers import guess_angles, guess_bonds, guess_dihedrals
from scipy.stats import norm


class compute_RDF:
    def __init__(self) -> None:
        pass

    @staticmethod
    def water_water(
        topology_file, dcd_file, save_location, ligand_selection="resname UNK"
    ):
        """the water to the water"""

        # Create a Universe object
        universe = mda.Universe(topology_file, dcd_file)

        # Select atoms for which you want to calculate RDF
        water = universe.select_atoms(f"type O and not {ligand_selection}")

        # Calculate RDF
        rdf = RDF.InterRDF(
            water, water, nbins=100, range=(0, 15.0), exclusion_block=(1, 1)
        )  # avoid computation to itself
        rdf.run()

        # print(rdf.results.bins)
        # print(rdf.results.rdf)

        # Plot RDF
        plt.plot(rdf.results.bins, rdf.results.rdf)
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("RDF")
        plt.title(f"Radial Distribution Function of water-water\nfor {dcd_file}")
        plt.savefig(f"{save_location}")
        # plt.show()
        plt.close()

    @staticmethod
    def _water_solute_base(
        topology_file, dcd_file, save_location, ligand_selection, atom_type="all"
    ):
        """water and selection of the ligand, used as a base for other functions"""

        # Create a Universe object
        universe = mda.Universe(topology_file, dcd_file)

        # Select atoms for which you want to calculate RDF
        water = universe.select_atoms(f"type O and not {ligand_selection}")
        if atom_type.lower().strip() != "all":
            ligand_selection = (
                f"type {atom_type.upper().strip()} and {ligand_selection}"
            )
        ligand = universe.select_atoms(f"{ligand_selection}")

        # Calculate RDF
        rdf = RDF.InterRDF(ligand, water, nbins=100, range=(0.0, 15.0))
        rdf.run()

        # Plot RDF
        plt.plot(rdf.results.bins, rdf.results.rdf)
        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("RDF")
        if not atom_type:
            atom_type = "all"
        plt.title(
            f"Radial Distribution Function of water-solute {atom_type}\nfor {dcd_file}"
        )
        plt.savefig(f"{save_location}")
        # plt.show()
        plt.close()

    def water_solute_all(
        topology_file, dcd_file, save_folder, ligand_selection="resname UNK"
    ):
        """water and all of the solute (define using the resname)"""

        compute_RDF._water_solute_base(
            topology_file, dcd_file, save_folder, ligand_selection, atom_type="all"
        )

    def water_solute_C(
        topology_file, dcd_file, save_folder, ligand_selection="resname UNK"
    ):
        """water and the C of the solute (define using the resname)"""

        compute_RDF._water_solute_base(
            topology_file, dcd_file, save_folder, ligand_selection, atom_type="type C"
        )

    def water_solute_H(
        topology_file, dcd_file, save_folder, ligand_selection="resname UNK"
    ):
        """water and the H of the solute (define using the resname)"""

        compute_RDF._water_solute_base(
            topology_file, dcd_file, save_folder, ligand_selection, atom_type="type H"
        )

    def water_solute_O(
        topology_file, dcd_file, save_folder, ligand_selection="resname UNK"
    ):
        """water and the O of the solute (define using the resname)"""

        compute_RDF._water_solute_base(
            topology_file, dcd_file, save_folder, ligand_selection, atom_type="type O"
        )

    # calculating the site specific RDF
    def water_atom(
        topology_file, dcd_file, save_location, ligand_selection="resname UNK"
    ):
        """water and each individual atom in the solute (define using the resname). Will plot together on the same plot."""

        # Create a Universe object
        universe = mda.Universe(topology_file, dcd_file)

        # Select atoms for which you want to calculate RDF
        water = universe.select_atoms(f"type O and not {ligand_selection}")
        ligand = universe.select_atoms(ligand_selection)
        atoms = [
            universe.select_atoms(f"index {atom.index} and {ligand_selection}")
            for atom in universe.select_atoms(ligand_selection)
        ]

        # Calculate RDF
        for atom in atoms:
            rdf = RDF.InterRDF(atom, water, nbins=100, range=(0, 15.0))
            rdf.run()
            # Plot RDF
            plt.plot(rdf.results.bins, rdf.results.rdf, label=atom[0].name)

        plt.xlabel("Distance (Angstrom)")
        plt.ylabel("RDF")
        plt.title(f"Radial Distribution Function of water-atom\nfor {dcd_file}")
        plt.legend(loc="center left")
        plt.savefig(f"{save_location}")
        plt.close()

    def single_water_single_atom(
        topology_file, dcd_file, save_folder=None, ligand_selection="resname UNK"
    ):
        """RDF for the single waters and all of the solute (define using the resname). Will plot each seperately."""

        plots_folder = f"{save_folder}/rdfs"
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Create a Universe object
        universe = mda.Universe(topology_file, dcd_file)

        # Select atoms for which you want to calculate RDF
        water = universe.select_atoms(f"type O and not {ligand_selection}")
        ags = [
            [universe.select_atoms(f"index {atom.index} and {ligand_selection}"), water]
            for atom in universe.select_atoms(ligand_selection)
        ]
        ss_rdf = RDF.InterRDF_s(
            universe, ags, nbins=100, range=(0, 15.0), norm="density", density=True
        )  # final RDF is over the average density of the selected atoms in the trajectory box
        ss_rdf.run()

        for idx, ag in enumerate(ags):
            # format
            # ss_rdf.rdf[idx][0][0]
            # first ags group (the first atom), the atom (the only atom in that atom group), the water molecule

            # if plotting each water
            for idx_w, wat in enumerate(ag[1]):
                plt.plot(ss_rdf.results.bins, ss_rdf.rdf[idx][0][idx_w])
                plt.xlabel("Distance (Angstrom)")
                plt.ylabel("RDF")
                plt.title(
                    f"RDF between {ag[0][0].name} of {ligand_selection} and {wat.name} {wat.resid}"
                )
                plt.savefig(
                    f"{plots_folder}/rdf_{ag[0][0].name}_{wat.name}_{wat.resid}.png"
                )
                plt.close

            # plot together on same plot
            # for idx_w, wat in enumerate(ag[1]):
            #     plt.plot(ss_rdf.results.bins, ss_rdf.rdf[idx][0][idx_w])
            # plt.xlabel('Distance (Angstrom)')
            # plt.ylabel('RDF')
            # plt.title(f'RDF between {ag[0][0].name} of {ligand_selection} and single waters')
            # plt.savefig(f"{save_folder}/rdf_{ag[0][0].name}_single_waters.png")
            # plt.close


def plot_dihedrals(
    topology_file, dcd_file, save_folder=None, ligand_selection="resname UNK"
):
    """plot the dihedrals of the solute (resname)"""
    plots_folder = f"{save_folder}/dihedrals"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    universe = mda.Universe(topology_file, dcd_file)
    ligand_atoms = universe.select_atoms(ligand_selection)

    bonds = guess_bonds(ligand_atoms, ligand_atoms.positions)

    universe.add_bonds(bonds)
    angles = guess_angles(universe.bonds)
    universe.add_angles(angles)
    dihedrals = guess_dihedrals(universe.angles)
    universe.add_dihedrals(dihedrals)

    # selection of atomgroups
    R = Dihedral(universe.dihedrals).run()

    # for all dihedrals
    # Get the dihedral angles
    dihedral_angles = R.angles

    plt.figure(figsize=(10, 6))

    legend_labels = []
    for idx, dihedral in enumerate(universe.dihedrals):
        label = f'Dihedral_{",".join(list(dihedral.atoms.names))}'
        plt.hist(dihedral_angles[:, idx], bins=50, density=True, alpha=0.5, label=label)
        legend_labels.append(label)

    plt.xlabel("Dihedral Angle (degrees)")
    plt.ylabel("Probability Density")
    plt.title("Probability Distribution Function for Dihedral Angles")
    legend = plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{plots_folder}/all_dihedrals.png")
    plt.close()

    # Extract colors used in the legend as dictionary keys
    legend_colors = {}
    for patch, label in zip(legend.get_patches(), legend_labels):
        legend_colors[label] = patch.get_facecolor()

    # Extract the dihedral angle data for one dihedral
    for idx, dihedral in enumerate(universe.dihedrals):
        label = f'Dihedral_{",".join(list(dihedral.atoms.names))}'
        # first part is the frames (rows), second part is the dihedral index (column)
        dihedral_data = R.angles[:, idx]

        # Plot the histogram of dihedral angles
        plt.hist(
            dihedral_data, bins=50, density=True, alpha=0.7, color=legend_colors[label]
        )

        # fit a probability density function to the histogram
        mu, sigma = norm.fit(dihedral_data)
        x = np.linspace(min(dihedral_data), max(dihedral_data), 100)
        pdf = norm.pdf(x, mu, sigma)
        plt.plot(x, pdf, "r-", linewidth=2)

        # Plot labels and title
        plt.xlabel("Dihedral Angle (degrees)")
        plt.ylabel("Probability Density")
        plt.title(f"Probability Density for {label}")
        # plt.show()
        plt.savefig(f"{plots_folder}/{label}.png")
        plt.close()


class plot_histograms:
    def __init__(self) -> None:
        pass

    # plotting histograms

    @staticmethod
    def _plot_stdout_histogram_base(txt_file, save_location, data_name, prop):
        df = pd.read_csv(txt_file, sep=",")
        data = df[f"{data_name}"]
        # fit a probability density function to the histogram
        mu, sigma = norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        pdf = norm.pdf(x, mu, sigma)

        plt.plot(x, pdf, "r-", linewidth=2, color="darkblue")
        plt.hist(data, bins=50, density=True, color="lightblue")
        plt.xlabel(f"{data_name}")
        plt.ylabel("Probability Density")
        plt.title(f"Histogram of {prop} for {txt_file}")
        plt.savefig(f"{save_location}")
        # plt.show()

    @staticmethod
    def plot_potential_energy(txt_file, save_location):
        plot_histograms._plot_stdout_histogram_base(
            txt_file,
            save_location,
            data_name="Potential Energy (kJ/mole)",
            prop="Potential Energy",
        )

    @staticmethod
    def plot_kinetic_energy(txt_file, save_location):  # using the output file density
        plot_histograms._plot_stdout_histogram_base(
            txt_file,
            save_location,
            data_name="Kinetic Energy (kJ/mole)",
            prop="Kinetic Energy",
        )  # mL is cm^3

    @staticmethod
    def plot_density(txt_file, save_location):  # using the output file density
        plot_histograms._plot_stdout_histogram_base(
            txt_file, save_location, data_name="Density (g/mL)", prop="Water Density"
        )  # mL is cm^3


class plot_energies:
    def __init__(self) -> None:
        pass

    # plotting with time
    @staticmethod
    def _plot_stdout_with_time_base(txt_file, save_location, data_name, prop):
        df = pd.read_csv(txt_file, sep=",")
        data = df[f"{data_name}"]
        steps = df[f'#"Step"']

        plt.plot(steps, data, "r-", linewidth=2, color="maroon")
        plt.xlabel(f"Steps")
        plt.ylabel(f"{data_name}")
        plt.title(f"Plot of {prop} for {txt_file}")
        # plt.savefig(f"{save_location}")
        plt.show()

    @staticmethod
    def plot_potential_energy(txt_file, save_location):
        plot_energies._plot_stdout_with_time_base(
            txt_file,
            save_location,
            data_name="Potential Energy (kJ/mole)",
            prop="Potential Energy",
        )

    @staticmethod
    def plot_kinetic_energy(txt_file, save_location):
        plot_energies._plot_stdout_with_time_base(
            txt_file,
            save_location,
            data_name="Kinetic Energy (kJ/mole)",
            prop="Kinetic Energy",
        )

    @staticmethod
    def plot_energy_all_windows(
        folder,
        save_location,
        prop="Kinetic Energy",
        base_name="stdout_prod_",
        no_windows=11,
    ):
        prop_dict = {
            "Kinetic Energy": "Kinetic Energy (kJ/mole)",
            "Potential Energy": "Potential Energy (kJ/mole)",
        }
        data_name = prop_dict[prop]

        plt.figure(figsize=(10, 6))

        cmap = plt.get_cmap("gnuplot")
        colors = [cmap(i) for i in np.linspace(0, 1, no_windows)]
        legend_labels = []
        for w in range(0, 11, 1):
            label = f"lambda {w}"
            txt_file = f"{folder}/{base_name}{w}.txt"
            df = pd.read_csv(txt_file, sep=",")
            data = df[f"{data_name}"]
            steps = df[f'#"Step"']
            x_axis = [r for r in range(1, len(steps) + 1, 1)]

            plt.plot(x_axis, data, linewidth=2, alpha=0.3, label=label, color=colors[w])
            legend_labels.append(label)
            plt.xlabel(f"Steps")
            plt.ylabel(f"{data_name}")
            plt.title(f"Plot of {prop} for {txt_file}")
            plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
            plt.tight_layout()
            plt.savefig(f"{save_location}")
        plt.show()
