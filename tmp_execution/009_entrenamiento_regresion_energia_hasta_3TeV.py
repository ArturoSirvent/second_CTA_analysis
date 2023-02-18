# %% [markdown]
#  # Vamos a hacer unos modelos para la predicción de energía.
#  Esto es casi igual que lo de la clasificación pero tenemos que hacerlo con las labels de energia.

# %%
#importamos librerias

import sys
sys.path.append('/home/asirvent/second_CTA_analysis/src/CTA-data-analisis-library/')

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
#enviroment variables
base_dir="/home/asirvent/second_CTA_analysis"
npy_final_dir=f"{base_dir}/datos/elementos_npy"
base_dir_elementos=f"{base_dir}/datos/elementos"
elements=['gamma', 'electron']


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
opciones_filtros=[
    [[16, 32], [64, 128], [128, 64], [64, 32]],
    [[32,64],[64,128],[128,64,32]],
    [[16,32],[32,64],[64,32,16]]]

opciones_filtros_last=[
    [20,10],[40,5],[10,5]
]


# %%
#vamos a llevar un registros del punto en el que se esta quedando el modelo
#voy a hacer una callback que guarde el último modelo, y con ello
#vamos a ir guardando los ultimo datos de la ejecucion y si hay un problema, retornamos a ese punto

#si tenemos que retomar el entrenamiento, vamos a indicar que modelo usar

file_number="009"
n=18 #repes de boostrap
#primer bucle para arquitecturas
for i,arch in enumerate(opciones_filtros):
    print(f"{i}: {arch} \n")
    modelo=models.model_multi_tel_energy(len_inputs=4,input_shapes=[(55,93,1)],filtros=arch,last_dense=opciones_filtros_last[i]) #no compila
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss="mse",metrics=["mae","mape"])
    with open(f"{base_dir}/automat/logs/{file_number}_data_control_energy.txt","a") as registro:
        registro.write(f"Con arquitectura: {arch} + {opciones_filtros_last[i]}") #, memoria {get_all_size(list(locals().items()))} Mb \n")

    #segundo_bucle para boostrap
    for k in range(n):
        #modificamos el learning rate
        if k == 7:
            modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss="mse",metrics=["mae","mape"])
        elif k == 12:
            modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss="mse",metrics=["mae","mape"])
        elif k == 15:
            modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),loss="mse",metrics=["mae","mape"])


        print(f"\n Boostrap {k+1} de {n}")#, memoria {get_all_size(list(locals().items()))} Mb \n")

        list_runs=new_create_main_list_runs([15,60],chose_runs)#new_create_main_list_runs([2,6,6,6,6,6,6],chose_runs)
        with open(f"{base_dir}/automat/logs/{file_number}_data_control_energy.txt","a") as registro:
            registro.write(f"Boostrap {k+1} de {n},runs: {list_runs}")#, memoria {get_all_size(list(locals().items()))} Mb \n")
        x_train_list,x_test_list,y_train_list,y_test_list=loaddata4use.load_dataset_energy(npy_final_dir,base_dir_elementos,elementos=['gamma', 'electron'],main_list_runs=list_runs,telescopios=[1,2,3,4],test_size=0.1,
            same_quant="same",verbose=True,fill=True,lower_energy_bound=0,upper_energy_bound=3)
        print(f"\n Variables cargadas")#, en memoria {get_all_size(list(locals().items()))} Mb \n")

        x_train_list=cambiar_ejes_lista(x_train_list)
        x_test_list=cambiar_ejes_lista(x_test_list)

        
        hist=modelo.fit(x=x_train_list,y=y_train_list,epochs=40, validation_data=(x_test_list,y_test_list),batch_size=64)
        del x_train_list,x_test_list,y_train_list,y_test_list
        #with open(f"{base_dir}/automat/logs/{file_number}_data_control_energy.txt","a") as registro:
        #    registro.write(f"Al borrar memoria nos quedan {get_all_size(list(locals().items()))} Mb \n")
        print(f"Split hecho")
        modelo.save(f"{base_dir}/modelos/{file_number}_modelo_filtro_{i}_en_boostrap_stage_{k+1}_energy.h5")
        with open(f"{base_dir}/modelos/performances/{file_number}_history_modelo_filtro_{i}_en_boostrap_stage_{k+1}_energy.pickle","wb") as pick:
            pickle.dump(hist.history,pick)    
        gc.collect()
    del modelo 
    gc.collect()




# %%






