"""Module with utility functions."""
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


def plot_lambda_schedule(
    lambda_schedule: dict, output_file: str = None, *args, **kwargs
) -> None:
    """
    Plot the lambda schedule.

    Parameters
    ----------
    lambda_schedule : dict
        Dictionary with the lambda values for the alchemical states.
        The keys of the dictionary are the lambda parameters and the values are lists of lambda values.
    output_file : str, optional, default=None
        The name of the output file to save the plot. If None, the plot is shown.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes

    for lambda_param, lambda_values in lambda_schedule.items():
        ax.plot(lambda_values, "-o", label=lambda_param, *args, **kwargs)

    ax.set_ylim(-0.05, 1.05)  # Set the y-axis limits
    ax.set_xlim(-0.5, len(lambda_values) - 0.5)  # Set the x-axis limits
    ax.set_xticks(list(range(0, len(lambda_values), 1)))
    ax.set_title(r"$\lambda$ schedule")
    ax.set_xlabel(r"$\lambda$ windows")
    ax.set_ylabel(r"$\lambda$ values")
    ax.grid(True)  # Add grid
    ax.legend()

    # Save the plot
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()
