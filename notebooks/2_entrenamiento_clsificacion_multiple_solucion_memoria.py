# %% [markdown]
# # DESCRIPCIÓN
# 
# Esto es lo mismo que hemos hecho antes de probar varias arquitecturas para la clasificación de tres categorias.  
# **- Pero ahora vamos a intentar mejorar la gestión de la memoria para que no se nos muera el proceso.**  
# **- La máquina tiene una GPU, vamos a intentar usarla.**  
# 

# %% [markdown]
# ## Carga de librerias y datos 
# Igual que antes (bucle_chustero...), pero ahora vamos a ver si tenemos gpu.

# %%
import sys
sys.path.append('../src/CTA-data-analisis-library/')

# %%
#cargamos librerias 
import os 
import subprocess
from datetime import datetime
import numpy as np 
import glob
import matplotlib.pyplot as plt
import tensorflow as tf 
import psutil
import re
import random
import shutil
import pickle
from numba import cuda
import gc

#propias
import unzipdata_and_first_treatments as manipulate
import loaddata4use
import model_creation_functions as models

# %%
tf.config.threading.set_inter_op_parallelism_threads(4)

# %% [markdown]
# Comprobamos el tema de gpu.

# %%
print(tf.config.experimental.get_memory_info("CPU:0")["current"]>>20)

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.get_visible_devices())


# %% [markdown]
# Pues no tenemos disponibles. Ala.

# %%
#enviroment variables
npy_final_dir="../datos/elementos_npy"
base_dir_elementos="../datos/elementos"
elements=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']

# %%
def runs_disponibles(npy_dir,elements):
    lista=[]
    for i in elements:
        npy_element_dir=os.path.join(npy_dir,"npy_"+i)
        runs=[int(re.search("run_([0-9]{3})_0\.npy",i).group(1)) for i in os.listdir(npy_element_dir) if re.search("run_([0-9]{3})_0\.npy",i)]
        lista.append(runs)
    return lista


def new_create_main_list_runs(number_runs_per_element,posibles_runs):
    #esto es aleatorio por defecto, porque es lo unico que necesito por ahora    
    final=[]
    for ind,lista_runs_element in enumerate(posibles_runs):
        final.append(random.sample(lista_runs_element,number_runs_per_element[ind]))
    return final

#tenemos que hacer un ligero cambio porque se estan cargando con los ejes cambiados
def cambiar_ejes_lista(lista):
    for i,j in enumerate(lista):
        lista[i]=np.swapaxes(j,1,2)
    return lista

# %%
#hacemos esto porque dan problemas las runs 2 y 3 para gamma por no tener correspondiente txt
chose_runs=runs_disponibles(npy_final_dir,elements)
chose_runs[0].remove(2)
chose_runs[0].remove(2)
chose_runs[0].remove(3)
chose_runs[0].remove(3)
chose_runs[0].remove(3)

# %%
opciones_filtros=[
    [[12,16,32],[64,128],[128,64,32]],
    [[12,16],[32,64],[64,12]],
    [[32,64],[128,64],[64,12]],
    [[32,64],[64,64],[32,16]],
    [[12,16,32],[32,64,128],[64,32,16]]
]

# %% [markdown]
# Pruebas sobre el tema de los grafos y los problemas de memoria.
# 

# %%
device = cuda.get_current_device()

with open("../modelos/2_data_control.txt","w") as registro:
    n=6 #repes de boostrap
    #primer bucle para arquitecturas
    for i,arch in enumerate(opciones_filtros):
        print(f"{i}: {arch} \n")
        modelo=models.model_multi_tel(classes=3,filtros=arch,last_dense=[20,5])
        modelo.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["acc","AUC","mean_squared_error"])
        registro.write(f"Con arquitectura: {arch} : \n")

        #segundo_bucle para boostrap
        for k in range(n):
            print(f"\n Boostrap {k+1} de {n}, y uso memoria CPU: {tf.config.experimental.get_memory_info('CPU:0')['current']>>20}mB\n")

            list_runs=new_create_main_list_runs([1,7,7,7,7,7,7],chose_runs)
            registro.write(f"Boostrap {k} de {n},runs: {list_runs},y uso memoria CPU: {tf.config.experimental.get_memory_info('CPU:0')['current']>>20}Mb \n")
            x_train_list,x_test_list,y_train_list,y_test_list=loaddata4use.load_dataset_completo(npy_final_dir,labels_asign=[0,1,2,2,2,2,2],elements=elements,
                                                                                                main_list_runs=list_runs,pre_name_folders="npy_",telescopes=[1,2,3,4],
                                                                                                test_size=0.1,same_quant="same",verbose=True,fill=True,categorical=True)
            x_train_list=cambiar_ejes_lista(x_train_list)
            x_test_list=cambiar_ejes_lista(x_test_list)

            
            hist=modelo.fit(x=x_train_list,y=y_train_list,epochs=5, validation_data=(x_test_list,y_test_list),batch_size=64)            
            gc.collect()
            del x_train_list,x_test_list,y_train_list,y_test_list
            modelo.save(f"../modelos/2_modelo_filtro_{i}_en_boostrap_stage_{k+1}.h5")
            with open(f"../modelos/performances/2_history_modelo_filtro_{i}_en_boostrap_stage_{k+1}.pickle","wb") as pick:
                pickle.dump(hist,pick)
        registro.write("\n")            
        gc.collect()
        del modelo 
        tf.keras.backend.clear_session()
        device.reset()
            


# %% [markdown]
# Nos sale con el cuda de numba que tenemos una GeForce, pero no nos sale de la otra manera con tf.  
# Hemos incluido una sentencias que resetean el backend de keras y tambien he metido los del mejor, antes estaban mal puestos.  

# %%


# %%



