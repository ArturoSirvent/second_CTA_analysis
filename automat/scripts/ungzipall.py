import sys
import os 
import subprocess

import numpy as np 
sys.path.append('/home/asirvent/second_CTA_analysis/src/CTA-data-analisis-library')
import unzipdata_and_first_treatments as manipulate
base_dir_elementos="/home/asirvent/SimTelArray_2022_05"

elementos=["proton","electron","helium","iron","nitrogen","silicon","gamma"]
manipulate.unzip_gunzip(base_dir_elementos,final_dir=None,elements=elementos)

