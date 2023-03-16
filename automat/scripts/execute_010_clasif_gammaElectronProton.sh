#!/bin/bash


#SBATCH --job-name=my_job_arturo_3TeV
#SBATCH --time=11:00:00
#SBATCH --mem=30GB
#SBATCH --partition guest
#SBATCH --output="execute_010_clasif_gammaElectronProton.out"

source /home/asirvent/aux_env/bin/activate 
python3 /home/asirvent/second_CTA_analysis/tmp_execution/010_entrenamiento_clasificacion_gammaElectronProton.py
