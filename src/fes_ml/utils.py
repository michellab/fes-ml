import openmm as _mm


def write_system_to_xml(system: _mm.System, filename: str) -> None:
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
        outfile.write(_mm.XmlSerializer.serialize(system))


def plot_lambda_schedule(lambda_schedule: dict, *args, **kwargs):
    """ """
    import matplotlib.pyplot as plt

    for lamba_param, lambda_vals in lambda_schedule.items():
        plt.plot(lambda_vals, "-o", label=lamba_param)

    plt.xlabel("Lambda window")
    plt.grid(True)
    plt.legend()
    plt.show()
