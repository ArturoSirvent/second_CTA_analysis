#este script esta para automatizar todo el tema del entrenamiento del entrenamiento, hacer que si tenemos un
#problema con la memoria o con la ejecución de un programa que lancemos, esto intente volverlo a correr
#este será un script ejecutado con $ nohup python -u script > nohup_master.out 2>&1 &
#pero a su vez, esto ejecutará comando con el nohup en el shell de la máquina, para retomar el entrenamiento
#desde el ultimo momento.

import subprocess
import os 
import sys