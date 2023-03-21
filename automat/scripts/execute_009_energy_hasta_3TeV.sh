#!/bin/bash


#SBATCH --job-name=my_job_arturo_3TeV
#SBATCH --time=0
#SBATCH --mem=30GB
#SBATCH --partition guest
#SBATCH --output="execute_009_energy_hasta_3TeV.out"

source /home/asirvent/python_env/bin/activate 
python3 /home/asirvent/second_CTA_analysis/tmp_execution/009_entrenamiento_regresion_energia_hasta_3TeV.py
