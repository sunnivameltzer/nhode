#!/bin/bash
#SBATCH --job-name=nhode_hnn_pot_seed0
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ---------------------------------------------------------------------
# Example Slurm script for running one NHODE training job.
#
# Before using:
#   1. Replace PROJECT_ROOT with the path to your local clone of this repository.
#   2. Replace CONDA_ENV with the name of your conda environment.
#   3. Submit from the repository root with:
#        sbatch docs/slurm_example.sh
# ---------------------------------------------------------------------

PROJECT_ROOT="$HOME/nhode"
CONDA_ENV="nhode"

. "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT/experiments/two_mass_two_spring_1d"        # Or replace "two_mass_two_spring_1d" with one of the other experiments

echo "==================================================="
echo "STARTING AT $(date)"
echo "==================================================="
echo ""

CONFIG_FILE="configs/hnn_pot/hnn_pot_0.yaml"

srun python -u run_training.py "$CONFIG_FILE"

echo ""
echo "==================================================="
echo "FINISHED AT $(date)"
echo "==================================================="