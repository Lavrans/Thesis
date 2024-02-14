#!/bin/sh
#SBATCH --account=ie-idi       # E.g. "ie-idi" if you belong to IE
#SBATCH --job-name=example_job
#SBATCH --time=0-00:15:00         # format: D-HH:MM:SS

#SBATCH --partition=CPUQ          # Asking for a GPU
#SBATCH --mem=5G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output.txt      # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'

#SBATCH --mail-user=lavranks@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
