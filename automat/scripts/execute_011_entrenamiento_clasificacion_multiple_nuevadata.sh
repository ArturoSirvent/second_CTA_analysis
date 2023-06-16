#!/bin/bash


#SBATCH --job-name=my_job_arturo_clasif
#SBATCH --time=11:00:00
#SBATCH --mem=30GB
#SBATCH --partition guest
#SBATCH --output="011_entrenamiento_clasificacion_multiple_nuevadata.out"

source /home/asirvent/aux_env/bin/activate 
python3 /home/asirvent/second_CTA_analysis/tmp_execution/011_entrenamiento_clasificacion_multiple_nuevadata.py
