#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/sling-nlp-showcase-%J.out
#SBATCH --error=logs/sling-nlp-showcase-%J.err
#SBATCH --job-name="SLING NLP showcase"

srun singularity exec --nv python-container.sif python few_shot.py
