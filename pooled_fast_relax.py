import pyrosetta
from pyrosetta import rosetta
import multiprocessing as mp
import argparse
import os
import math
import random

## pyrosetta/database/chemical/residue_type_sets/fa_standard/merge_residue_behaviors.txt was modified to prevent merging the ACE and NME residues with the standard amino acids.
## Will still have to look at the hydrogen naming issue. might not be a problem, as it can be fixed later on in the openmm portion of the pipeline.

# Might be a good idea to do one round of fast relax, and use the generated structure as the input for the next round of fast relax. This way, the structures will be more similar to the original structure, and the Energy values will be more meaningful (Now conformer energy values are distant from the original value).

# FastRelax protocol with optional cycles parameter
def run_fastrelax(seed, input_pdb, output_pdb, cycles=20):
    # Initialize PyRosetta with a random seed
    pyrosetta.init(f"-constant_seed -jran {seed} -ignore_waters true")

    # Load the pose
    pose = pyrosetta.pose_from_pdb(input_pdb)

    # Create and configure the FastRelax object
    fastrelax = rosetta.protocols.relax.FastRelax()
    scorefxn = pyrosetta.get_fa_scorefxn()
    fastrelax.set_scorefxn(scorefxn)

    # Set the number of cycles for the relaxation process
    fastrelax.max_iter(cycles)  # Use the cycles parameter to control sampling depth

    # Apply FastRelax to the pose
    fastrelax.apply(pose)

    # Save the relaxed structure
    pose.dump_pdb(output_pdb)
    return f"Relaxation with seed {seed} and {cycles} cycles completed."

def run_parallel_relaxation(input_pdb, output_dir, n_relax, n_processes, cycles):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Total number of batches
    n_batches = math.ceil(n_relax / n_processes)

    # Generate a list of random seeds
    all_seeds = random.sample(range(1, 1000000), n_relax)

    for batch in range(n_batches):
        # Determine start and end of the current batch
        start = batch * n_processes
        end = min(start + n_processes, n_relax)

        # Prepare tasks for the current batch
        tasks = [(all_seeds[i], input_pdb, os.path.join(output_dir, f"relaxed_structure_{i}.pdb"), cycles) 
                 for i in range(start, end)]

        # Run the current batch in parallel
        with mp.Pool(processes=n_processes) as pool:
            results = pool.starmap(run_fastrelax, tasks)

        # Print results of the current batch
        for res in results:
            print(res)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Run multiple FastRelax simulations in parallel.")
    parser.add_argument("-i", "--input", required=True, help="Input PDB file")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for relaxed structures")
    parser.add_argument("-n", "--n_relax", type=int, required=True, help="Number of FastRelax runs to perform")
    parser.add_argument("-p", "--n_processes", type=int, required=True, help="Number of parallel processes to use")
    parser.add_argument("-c", "--cycles", type=int, default=20, help="Number of cycles for FastRelax (default: 20)")

    args = parser.parse_args()

    # Run parallel relaxation with the specified number of cycles
    run_parallel_relaxation(args.input, args.output_dir, args.n_relax, args.n_processes, args.cycles)

if __name__ == "__main__":
    main()