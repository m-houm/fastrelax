
####################################################################################################

import pyrosetta
from pyrosetta import rosetta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compare_structures(input_pdb, output_dir, plot=False, initialize=True, interface=False, interface1_chains=None, interface2_chains=None):
    """
    Compare generated conformers with the original structure.

    Args:
        input_pdb (str): Path to the original PDB file.
        output_dir (str): Directory containing generated conformer PDB files.
        plot (bool): If True, plot distance metrics against energy metrics.
        initialize (bool): If True, initialize PyRosetta. Default is True.
        interface (bool): If True, calculate interface energy for the given chain pairs.
        interface1_chains (list of str): List of chain IDs for interface 1.
        interface2_chains (list of str): List of chain IDs for interface 2.

    Returns:
        pd.DataFrame: DataFrame containing all comparison metrics.
    """

    # Initialize PyRosetta only if needed
    if initialize:
        pyrosetta.init()

    # Load the original input structure
    original_pose = pyrosetta.pose_from_pdb(input_pdb)

    # Get the list of generated PDB files from the output directory
    generated_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.pdb')]

    # Prepare lists to store metrics for comparison
    data = []

    # Score function for energy calculations
    scorefxn = pyrosetta.get_fa_scorefxn()

    for generated_pdb in generated_files:
        # Load the generated structure
        generated_pose = pyrosetta.pose_from_pdb(generated_pdb)

        # Calculate RMSDs
        full_rmsd = rosetta.core.scoring.CA_rmsd(original_pose, generated_pose)  # CA RMSD (all atoms)
        backbone_rmsd = rosetta.core.scoring.bb_rmsd(original_pose, generated_pose)  # Backbone RMSD
        all_atom_rmsd = rosetta.core.scoring.all_atom_rmsd(original_pose, generated_pose)  # All-atom RMSD
        heavy_atom_rmsd = rosetta.core.scoring.rms_at_corresponding_heavy_atoms(original_pose, generated_pose)  # Heavy Atom RMSD

        # Calculate Solvent Accessible Surface Area (SASA)
        sasa_metric = rosetta.core.scoring.sasa.SasaCalc()
        original_sasa = sasa_metric.calculate(original_pose)
        generated_sasa = sasa_metric.calculate(generated_pose)
        sasa_ratio = generated_sasa / original_sasa if original_sasa != 0 else np.nan

        # Calculate HBond Energy using score function
        hbond_energy = scorefxn.score_by_scoretype(generated_pose, rosetta.core.scoring.ScoreType.hbond_sc) + \
                       scorefxn.score_by_scoretype(generated_pose, rosetta.core.scoring.ScoreType.hbond_bb_sc)

        # Calculate energetics
        original_energy = scorefxn(original_pose)
        generated_energy = scorefxn(generated_pose)
        energy_ratio = generated_energy / original_energy if original_energy != 0 else np.nan

        interaction_energy = scorefxn.score_by_scoretype(generated_pose, rosetta.core.scoring.ScoreType.fa_atr) + \
                            scorefxn.score_by_scoretype(generated_pose, rosetta.core.scoring.ScoreType.fa_rep) + \
                            scorefxn.score_by_scoretype(generated_pose, rosetta.core.scoring.ScoreType.fa_sol)

        # Calculate interface score if specified
        interface_score = np.nan  # Default value
        if interface and interface1_chains and interface2_chains:
            # Create a selection string for the interface
            interface_str = f"{','.join(interface1_chains)}_{','.join(interface2_chains)}"
            interface_analyzer = rosetta.protocols.analysis.InterfaceAnalyzerMover(interface_str)
            interface_analyzer.apply(generated_pose)
            interface_score = interface_analyzer.get_interface_dG()

        # Append results to data list
        data.append({
            'filename': os.path.basename(generated_pdb),
            'full_rmsd': full_rmsd,
            'backbone_rmsd': backbone_rmsd,
            'all_atom_rmsd': all_atom_rmsd,
            'heavy_atom_rmsd': heavy_atom_rmsd,
            'sasa_ratio': sasa_ratio,
            'hbond_energy': hbond_energy,
            'original_energy': original_energy,
            'generated_energy': generated_energy,
            'energy_ratio': energy_ratio,
            'interaction_energy': interaction_energy,
            'interface_score': interface_score
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Print DataFrame
    print(df)

    # If plot is True, create plots for distance metrics against energy metrics
    if plot:
        distance_metrics = ['full_rmsd', 'backbone_rmsd', 'all_atom_rmsd', 'heavy_atom_rmsd', 'sasa_ratio']
        energy_metrics = ['generated_energy', 'energy_ratio', 'interaction_energy', 'hbond_energy', 'interface_score']

        for distance_metric in distance_metrics:
            for energy_metric in energy_metrics:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df, x=distance_metric, y=energy_metric)
                plt.title(f"{distance_metric} vs {energy_metric}")
                plt.xlabel(distance_metric.replace('_', ' ').title())
                plt.ylabel(energy_metric.replace('_', ' ').title())
                plt.grid(True)
                plt.show()

    return df