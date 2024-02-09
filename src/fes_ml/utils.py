import openmm as mm


def write_system_to_xml(system: mm.System, filename: str) -> None:
    """
    Write the System to an XML file.

    Parameters
    ----------
    system : openmm.System
        The System to write.
    filename : str
        The name of the file to write.
    """
    with open(filename, "w") as outfile:
        outfile.write(mm.XmlSerializer.serialize(system))
