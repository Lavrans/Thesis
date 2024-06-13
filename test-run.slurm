#!/bin/sh
#SBATCH --account=ie-idi       # E.g. "ie-idi" if you belong to IE
#SBATCH --job-name=test_run
#SBATCH --time=2-00:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100"
#SBATCH --mem=32G                 # Asking for 32GB RAM
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

module load Anaconda3/2022.10
conda activate venv
python train_lstm.py
