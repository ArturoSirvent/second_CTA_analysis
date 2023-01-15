# %% [markdown]
# # Vamos a hacer unos modelos para la predicción de energía.  
# 
# 
# Igual que el 003 pero vamos a aprovechar los modelos que ya tenemos y los vamos a finetunear.

# %%
#importamos librerias

import sys
sys.path.append('../src/CTA-data-analisis-library/')
import os 
import subprocess
from datetime import datetime
import numpy as np 
from  glob import glob
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

# %%
#enviroment variables
npy_final_dir="../datos/elementos_npy"
base_dir_elementos="../datos/elementos"
elements=['gamma', 'electron']
modelos_dir="../modelos"
lista_modelos_to_use=glob(f"{modelos_dir}/*stage_7.h5")

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

def get_all_size(local_vars):
    #local_vars = list(locals().items())
    total=0
    for var, obj in local_vars:
        total+=sys.getsizeof(obj)
    total= total >> 20
    return total # its in Mb

# %%
chose_runs=runs_disponibles(npy_final_dir,elements)
chose_runs[0].remove(2)
chose_runs[0].remove(2)
chose_runs[0].remove(3)
chose_runs[0].remove(3)
chose_runs[0].remove(3)

# %%
#vamos a llevar un registros del punto en el que se esta quedando el modelo
#voy a hacer una callback que guarde el último modelo, y con ello
#vamos a ir guardando los ultimo datos de la ejecucion y si hay un problema, retornamos a ese punto

#si tenemos que retomar el entrenamiento, vamos a indicar que modelo usar

device = cuda.get_current_device()
file_number="004"
n=18 #repes de boostrap
numruns=[2,14]
#primer bucle para arquitecturas
for i,modelo_name in enumerate(lista_modelos_to_use):
    tf.keras.backend.clear_session()
    modelo=tf.keras.models.load_model(modelo_name)

    print(f"Modelo: {i}\n")

    #parte de cargar el modelo
    modelo2=tf.keras.models.Model(inputs=modelo.input,outputs=tf.keras.layers.Dense(1,name="asdf")(modelo.output))

    count=0
    for i in modelo2.layers:
        if str(type(i))=="<class 'keras.layers.core.dense.Dense'>":
            count+=1


    for indx,layer in enumerate(modelo2.layers):
        if indx > len(modelo2.layers)-1-count:
            pass
        else:
            layer.trainable=False



    modelo2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss="mse",metrics=["mae","mape"])
    with open(f"../automat/logs/{file_number}_data_control_energy.txt","a") as registro:
        registro.write(f"Con modelo: {i}, y uso de CPU {tf.config.experimental.get_memory_info('CPU:0')['current']>>20}Mb, y memoria {get_all_size(list(locals().items()))} Mb \n")

    #segundo_bucle para boostrap
    for k in range(n):
        #modificamos el learning rate
        if k == 4:
            modelo2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss="mse",metrics=["mae","mape"])
        elif k == 10:
            modelo2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss="mse",metrics=["mae","mape"])
        elif k == 12:
            modelo2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),loss="mse",metrics=["mae","mape"])
            numruns=[1,8]
            #y ponemos todo trainable
            for layer in modelo2.layers:
                layer.trainable=True



        print(f"\n Boostrap {k+1} de {n}, y uso de CPU {tf.config.experimental.get_memory_info('CPU:0')['current']>>20}Mb, y memoria {get_all_size(list(locals().items()))} Mb \n")

        list_runs=new_create_main_list_runs(numruns,chose_runs)#new_create_main_list_runs([2,6,6,6,6,6,6],chose_runs)
        with open(f"../automat/logs/{file_number}_data_control_energy.txt","a") as registro:
            registro.write(f"Boostrap {k+1} de {n},runs: {list_runs}, y uso de CPU {tf.config.experimental.get_memory_info('CPU:0')['current']>>20}Mb, y memoria {get_all_size(list(locals().items()))} Mb \n")
        x_train_list,x_test_list,y_train_list,y_test_list=loaddata4use.load_dataset_energy(npy_final_dir,base_dir_elementos,elementos=['gamma', 'electron'],main_list_runs=list_runs,telescopios=[1,2,3,4],test_size=0.01,same_quant="same",verbose=True,fill=True)
        print(f"\n Despues de cargar las variables, en memoria {get_all_size(list(locals().items()))} Mb \n")

        x_train_list=cambiar_ejes_lista(x_train_list)
        x_test_list=cambiar_ejes_lista(x_test_list)

        
        hist=modelo2.fit(x=x_train_list,y=y_train_list,epochs=7, validation_data=(x_test_list,y_test_list),batch_size=32)
        del x_train_list,x_test_list,y_train_list,y_test_list
        gc.collect()
        with open(f"../automat/logs/{file_number}_data_control_energy.txt","a") as registro:
            registro.write(f"Al borrar memoria nos quedan {get_all_size(list(locals().items()))} Mb \n")
        print(f"\n Al borrar memoria nos quedan {get_all_size(list(locals().items()))} Mb \n")
        modelo2.save(f"../modelos/{file_number}_modelo_filtro_{i}_en_boostrap_stage_{k+1}_energy.h5")
        with open(f"../modelos/performances/{file_number}_history_modelo_filtro_{i}_en_boostrap_stage_{k+1}_energy.pickle","wb") as pick:
            pickle.dump(hist,pick)    
    del modelo2 
    gc.collect()
    tf.keras.backend.clear_session()
    device.reset()

        


# %%



