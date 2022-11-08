import sys
import os 
import subprocess

import numpy as np 
sys.path.append('./CTA-data-analisis-library/')
import unzipdata_and_first_treatments as manipulate

npy_final_dir="/home/arturoSF/datos/elementos_npy"
base_dir_elementos="/home/arturoSF/datos/elementos"

elementos=["proton","electron","helium","iron","nitrogen","silicon","gamma"]
manipulate.dt_2_npy(base_dir_elementos,npy_final_dir,elementos,verbose=True)

