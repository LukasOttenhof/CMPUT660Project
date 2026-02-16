#!/bin/bash
#SBATCH --job-name=code_complexity

#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=code_complexity_%j.out
#SBATCH --error=code_complexity_%j.err

echo "Job submitted at: $(date)"

echo "Job started at: $(date)"
START_TIME=$(date +%s)
module load gcc
module load arrow
# Activate your virtual environment
source /home/lottenho/660_pro/venv/bin/activate

# Optional: print Python path to confirm venv is active
which python
python --version

# Run your script
python build_code_complexity.py

END_TIME=$(date +%s)
echo "Job ended at: $(date)"
echo "Execution time (seconds): $((END_TIME - START_TIME))"
