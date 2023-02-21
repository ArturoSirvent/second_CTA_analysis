#!/bin/bash


#SBATCH --job-name=my_job_arturo_3TeV
#SBATCH --time=11:00:00
#SBATCH --mem=30GB
#SBATCH --partition guest
#SBATCH --output="execute_009_ANALYSIS_energy_hasta_3TeV_v2.out"

source /home/asirvent/python_env/bin/activate 
python3 /home/asirvent/second_CTA_analysis/tmp_execution/009_analisis_resultados_energia_hasta_3TeV_v2.py
