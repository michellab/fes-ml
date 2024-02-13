"""Module with functions to manage the creation of alchemical systems."""
import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit


_NON_BONDED_METHODS = {
    0: _app.NoCutoff,
    1: _app.CutoffNonPeriodic,
    2: _app.CutoffPeriodic,
    3: _app.Ewald,
    4: _app.PME,
}

def add_LJ_softcore(system, alchemical_atoms, lambda_lj=1.0):
    """
    Add a CustomNonbondedForce to the System to compute the softcore
    Lennard-Jones and Coulomb interactions between the ML and MM region.

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    alchemical_atoms : list of int
        The indices of the atoms to model with the ML potential.
    lambda_lj : float, optional, default=1.0
        The softcore parameter for the Lennard-Jones interactions.

    Returns
    -------
    system : openmm.System
        The modified System with the softcore potentials added.
    """
    forces = {force.__class__.__name__: force for force in system.getForces()}
    nb_force = forces["NonbondedForce"]

    # Define the softcore Lennard-Jones energy function
    energy_function = (
        f"{lambda_lj}*4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
    )
    energy_function += (
        f"reff_sterics = sigma*(0.5*(1.0-{lambda_lj}) + (r/sigma)^6)^(1/6);"
    )
    energy_function += (
        "sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);"
    )

    # Create a CustomNonbondedForce to compute the softcore Lennard-Jones and Coulomb interactions
    soft_core_force = _mm.CustomNonbondedForce(energy_function)

    if _NON_BONDED_METHODS[nb_force.getNonbondedMethod()] in [_NON_BONDED_METHODS[3], _NON_BONDED_METHODS[4]]:
        print("The softcore Lennard-Jones interactions are not implemented for Ewald or PME")
        print("The nonbonded method will be set to CutoffPeriodic")
        soft_core_force.setNonbondedMethod(2)
    else:
        soft_core_force.setNonbondedMethod(nb_force.getNonbondedMethod())

    soft_core_force.setCutoffDistance(nb_force.getCutoffDistance())
    soft_core_force.setUseSwitchingFunction(nb_force.getUseSwitchingFunction())
    soft_core_force.setSwitchingDistance(nb_force.getSwitchingDistance())

    if abs(lambda_lj) < 1e-8:
        # Cannot use long range correction with a force that does not depend on r
        soft_core_force.setUseLongRangeCorrection(False)
    else:
        soft_core_force.setUseLongRangeCorrection(nb_force.getUseDispersionCorrection())

    # https://github.com/openmm/openmm/issues/1877
    # Set the values of sigma and epsilon by copying them from the existing NonBondedForce
    # Epsilon will always be 0 for the ML atoms as the LJ 12-6 interaction is removed
    soft_core_force.addPerParticleParameter("sigma")
    soft_core_force.addPerParticleParameter("epsilon")
    for index in range(system.getNumParticles()):
        [charge, sigma, epsilon] = nb_force.getParticleParameters(index)
        soft_core_force.addParticle([sigma, epsilon])
        if index in alchemical_atoms:
            # Remove the LJ 12-6 interaction
            nb_force.setParticleParameters(index, charge, sigma, 0.0)

    # Set the custom force to occur between just the alchemical particle and the other particles
    mm_atoms = set(range(system.getNumParticles())) - set(alchemical_atoms)
    soft_core_force.addInteractionGroup(alchemical_atoms, mm_atoms)

    # Add the CustomNonbondedForce to the System
    system.addForce(soft_core_force)

    return system

def scale_charges(system, alchemical_atoms, lambda_q=1.0):
    """
    Scale the charges of the alchemical atoms in the System.

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    alchemical_atoms : list of int
        The indices of the atoms to scale.
    lambda_q : float, optional, default=1.0
        The value to scale the charges by.

    Returns
    -------
    system : openmm.System
        The modified System with the charges scaled.
    """
    forces = {force.__class__.__name__: force for force in system.getForces()}
    nb_force = forces["NonbondedForce"]

    for index in range(system.getNumParticles()):
        [charge, sigma, epsilon] = nb_force.getParticleParameters(index)
        if index in alchemical_atoms:
            # Scale the charge and remove the LJ 12-6 interaction
            nb_force.setParticleParameters(index, charge * lambda_q, sigma, epsilon)

    return system

def add_alchemical_ML_region(
    ml_potential,
    topology,
    system,
    alchemical_atoms,
    interpolate=True,
    lambda_interpolate=1.0,
):
    """
    Create an alchemical System that is partly modeled with a ML potential and partly
    with a conventional force field.

    Parameters
    ----------
    ml_potential : openmmml.mlpotential.MLPotential
        The ML potential to use.
    topology : openmm.app.Topology
        The Topology of the System.
    system : openmm.System
        The System to modify. A copy of the System will be returned.
    alchemical_atoms : list of int
        The indices of the atoms to model with the ML potential.
    interpolate : bool, optional, default=True
        If True, the System will include Forces to compute the energy both with the
        conventional force field and with this potential, and to smoothly interpolate
        between them.  If False, the System will include only the Forces to compute
        the energy with this potential.
    lambda_interpolate : float, optional, default=1.0
        The value of the global parameter "lambda_interpolate" to use for the ML potential.

    Returns
    -------
    system : openmm.System
        The modified System.

    Notes
    -----
    The CustomCVForce defines a global parameter called "lambda_interpolate" that interpolates
    between the two potentials.  When lambda_interpolate=0, the energy is computed entirely with
    the conventional force field.  When lambda_interpolate=1, the energy is computed entirely with
    the ML potential.  You can set its value by calling setParameter() on the Context.
    """
    system = ml_potential.createMixedSystem(
        topology, system, alchemical_atoms, interpolate=interpolate
    )

    # Get the CustomCVForce that interpolates between the two potentials and set its global parameter
    # TODO: generalise this to work in cases where there are multiple CustomCVForces
    forces = {force.__class__.__name__: force for force in system.getForces()}
    cv_force = forces["CustomCVForce"]
    cv_force.setGlobalParameterDefaultValue(0, lambda_interpolate)

    return system

def _add_intramolecular_nonbonded_exceptions(self, system, alchemical_atoms):
    """
    Add exceptions to the NonbondedForce and CustomNonbondedForces
    to prevent the alchemical atoms from interacting as these interactions
    are already taken into account by the CustomBondForce.

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    alchemical_atoms : list of int
        The indices of the alchemical atoms.

    Returns
    -------
    system : openmm.System
        The modified System.
    """
    atom_list = list(alchemical_atoms)
    for force in system.getForces():
        if isinstance(force, _mm.NonbondedForce):
            for i in range(len(atom_list)):
                for j in range(i):
                    force.addException(atom_list[i], atom_list[j], 0, 1, 0, True)
        elif isinstance(force, _mm.CustomNonbondedForce):
            existing = set(
                tuple(force.getExclusionParticles(i))
                for i in range(force.getNumExclusions())
            )
            for i in range(len(atom_list)):
                a1 = atom_list[i]
                for j in range(i):
                    a2 = atom_list[j]
                    if (a1, a2) not in existing and (a2, a1) not in existing:
                        force.addExclusion(a1, a2, True)
    return system

def _add_intramolecular_nonbonded_forces(system, alchemical_atoms, reaction_field=False):
    """
    Add the intramolecular nonbonded forces as a CustomBondForce to the System.

    Parameters
    ----------
    system : openmm.System
        The System to modify.
    alchemical_atoms : list of int
        The indices of the alchemical atoms.
    reaction_field : bool, optional, default=False
        If True, the energy expression will include a reaction field term.
    
    Returns
    -------
    system : openmm.System
        The modified System.

    Notes
    -----
    This code is heavily inspired on REF.
    """
    forces = {force.__class__.__name__: force for force in system.getForces()}
    nb_force = forces["NonbondedForce"]

    if reaction_field:
        # Read: https://github.com/openmm/openmm/issues/3281
        # Read: http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html?highlight=cutoffperiodic
        cutoff = nb_force.getCutoffDistance()
        eps_solvent = nb_force.getReactionFieldDielectric()
        krf = (1/ (cutoff**3)) * (eps_solvent - 1) / (2*eps_solvent + 1)
        crf = (1/ cutoff) * (3* eps_solvent) / (2*eps_solvent + 1)
        energy_expression = "138.9354558466661*chargeProd*(1/r + krf*r*r - crf) + 4*epsilon*((sigma/r)^12-(sigma/r)^6);"
        energy_expression += f"krf = {krf.value_in_unit(_unit.nanometer**-3)};"
        energy_expression += f"crf = {crf.value_in_unit(_unit.nanometer**-1)}"
    else:
        energy_expression = "138.9354558466661*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)"

    internal_nonbonded = _mm.CustomBondForce(
        energy_expression
    )
    internal_nonbonded.addPerBondParameter("chargeProd")
    internal_nonbonded.addPerBondParameter("sigma")
    internal_nonbonded.addPerBondParameter("epsilon")
    numParticles = system.getNumParticles()
    atomCharge = [0] * numParticles
    atomSigma = [0] * numParticles
    atomEpsilon = [0] * numParticles
    for i in range(numParticles):
        charge, sigma, epsilon = nb_force.getParticleParameters(i)
        atomCharge[i] = charge
        atomSigma[i] = sigma
        atomEpsilon[i] = epsilon
    exceptions = {}
    for i in range(nb_force.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
        exceptions[(p1, p2)] = (chargeProd, sigma, epsilon)
    for p1 in alchemical_atoms:
        for p2 in alchemical_atoms:
            if p1 == p2:
                break
            if (p1, p2) in exceptions:
                chargeProd, sigma, epsilon = exceptions[(p1, p2)]
            elif (p2, p1) in exceptions:
                chargeProd, sigma, epsilon = exceptions[(p2, p1)]
            else:
                chargeProd = atomCharge[p1] * atomCharge[p2]
                sigma = 0.5 * (atomSigma[p1] + atomSigma[p2])
                epsilon = _unit.sqrt(atomEpsilon[p1] * atomEpsilon[p2])
            if chargeProd._value != 0 or epsilon._value != 0:
                internal_nonbonded.addBond(p1, p2, [chargeProd, sigma, epsilon])

    system.addForce(internal_nonbonded)

    return system

def alchemify(
    self,
    alchemical_atoms,
    lambda_i=None,
    lambda_u=None,
    lambda_x=None,
    ml_potential=None,
    topology=None,
):
    """
    Alchemify the system.

    Parameters
    ----------
    alchemical_atoms : list of int
        The indices of the atoms to model with the ML potential.
    lambda_i : float, optional, default=None
        The value of the global parameter "lambda_interpolate" to use for the ML potential.
    lambda_u : float, optional, default=None
        The value of the parameter "lambda_lj" to use for the softned Lennard-Jones interactions.
    lambda_x : float, optional, default=None
        The value of the parameter "lambda_q" to use for the softened Coulomb interactions.
    ml_potential : str, optional, default=None
        The name of the ML potential to use.  If None, "ani2x" will be used.
    topology : openmm.app.Topology
        The Topology of the System.

    Returns
    -------
    system : openmm.System
        The alchemified System.
    """
    if lambda_i is not None:
        try:
            from openmmml.mlpotential import MLPotential
        except ImportError:
            raise ImportError(
                "The openmmml package is required to use the ML potential"
            )

        if topology is None:
            raise ValueError(
                "The topology must be provided if the ML potential is not None"
            )

        if not isinstance(topology, _app.Topology):
            raise ValueError(
                "The topology must be an instance of openmm.app.Topology"
            )

        if ml_potential is None:
            ml_potential = "ani2x"

        ml_potential = MLPotential(ml_potential)
        system = add_alchemical_ML_region(
            ml_potential,
            topology,
            system,
            alchemical_atoms,
            interpolate=True,
            lambda_interpolate=lambda_i,
        )

    if (lambda_u is not None or lambda_x is not None) and lambda_i is None:
        system = _add_intramolecular_nonbonded_forces(
            system, alchemical_atoms
        )
        _add_intramolecular_nonbonded_exceptions(
            system, alchemical_atoms
        )
    if lambda_u is not None:
        system = add_LJ_softcore(system, alchemical_atoms, lambda_u)

    if lambda_x is not None:
        system = scale_charges(system, alchemical_atoms, lambda_x)
        
    return system



