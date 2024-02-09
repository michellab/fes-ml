import BioSimSpace as BSS

molecule = BSS.Parameters.gaff("c1ccccc1").getMolecule()
box, angles = BSS.Box.cubic(35 * BSS.Units.Length.angstrom)
solvated = BSS.Solvent.tip3p(molecule=molecule, box=box)

BSS.IO.saveMolecules("benzene", solvated, ["pdb", "prm7", "rst7"])

# f = sr.save(mols.trajectory(), f"alanine_dipeptide_plain_MACE_MM_MD_{suffix}", format=["DCD"])
