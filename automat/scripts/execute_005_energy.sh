#!/bin/bash
#SBATCH --job-name=my_job_arturo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=15GB
source /home/asirvent/python_env/bin/activate 
python3 /home/asirvent/second_CTA_analysis/tmp_execution/005_entrenamiento_regresion_energia.py
