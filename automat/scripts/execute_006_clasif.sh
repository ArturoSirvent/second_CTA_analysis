#!/bin/bash
#SBATCH --job-name=my_job_arturo
#SBATCH --time=0
#SBATCH --mem=30GB
#SBATCH --partition guest

source /home/asirvent/python_env/bin/activate 
python3 /home/asirvent/second_CTA_analysis/tmp_execution/006_entrenamiento_clasificacion_multiple_solucion_memoria.py
