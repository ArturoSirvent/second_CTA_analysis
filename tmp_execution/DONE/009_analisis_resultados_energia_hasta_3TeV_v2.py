# %% [markdown]
# Para cargar los datos en memria voy a usar funciones básicas.

# %%
import numpy as np
import matplotlib.pyplot as plt 
import pickle

#importamos librerias
import os 
import sys
sys.path.append('/home/asirvent/second_CTA_analysis/src/CTA-data-analisis-library/')

import subprocess

from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
import pandas as pd
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
BASE_DIR="/home/asirvent/second_CTA_analysis"
"""
# %%
fig0=plt.figure(figsize=(12,10))

for j in [0,1,2]:
    plt.subplot(3,1,j+1)
    aux1=[]
    aux3=[]
    for i in range(1,19):
        with open(f"{BASE_DIR}/modelos/performances/009_history_modelo_filtro_{j}_en_boostrap_stage_{i}_energy.pickle","rb") as fil:
            hist=pickle.load(fil)
            aux1.append(hist["loss"])
            aux3.append(hist["val_loss"])
    aux2=[j for i in aux1 for j in i]
    aux4=[j for i in aux3 for j in i]

    plt.plot(aux2,label="loss")
    plt.plot(aux4,label="val_loss")
    plt.title(f"Train loss vs validation loss (arch: {j})")
    plt.tight_layout()
    plt.legend()
    plt.grid()

plt.savefig(f"{BASE_DIR}/results/009_energy_training_losses.png")
plt.clf()
plt.close(fig0)
"""
# %% [markdown]
# ## Carga de datos de test

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
#enviroment variables
npy_final_dir=f"{BASE_DIR}/datos/elementos_npy_test"
base_dir_elementos=f"{BASE_DIR}/datos/elementos"
elements=['gamma', 'electron']

# %%
chose_runs=runs_disponibles(npy_final_dir,elements)


# %%
list_runs=new_create_main_list_runs([6,17],chose_runs)#new_create_main_list_runs([2,6,6,6,6,6,6],chose_runs)

x_train_list,x_test_list,y_train_list,y_test_list,element_shapes=loaddata4use.load_dataset_energy(npy_final_dir,base_dir_elementos,elementos=['gamma', 'electron'],
                                                        main_list_runs=list_runs,telescopios=[1,2,3,4],test_size=0,return_shapes=True,
                                                        same_quant="same",verbose=True,fill=True,lower_energy_bound=0,
                                                        upper_energy_bound=3)
x_train_list=cambiar_ejes_lista(x_train_list)
#x_test_list=cambiar_ejes_lista(x_test_list)


# %%
#plt.hist(y_train_list)

# %%
#cargamos el ultimo modelo de los dos filtros:
modelos=[tf.keras.models.load_model(name) for name in glob(f"{BASE_DIR}/modelos/009_modelo_filtro_*_en_boostrap_stage_18_energy.h5")]


# %%
y_pred_all=[]
total_len=x_train_list[1].shape[0]
n=450
cicles=int(np.ceil(total_len/n))
for i in range(cicles):
    print("Ciclo",i)
    if (n*i+n)>=total_len:
        final=total_len
    else:
        final=n*i+n

    x_aux=[ x[(n*i):final,:,:,:] for x in x_train_list]
    print("Shape",x_aux[0].shape)
    y_pred=[model(x_aux).numpy() for model in modelos]
    if i==0:
        y_pred_all=y_pred
    else:
        for j in range(len(y_pred_all)):
            y_pred_all[j]=np.concatenate([y_pred_all[j],y_pred[j]],axis=0)



# %%
# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                      # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results["y_pred"]=yhat
    results['determination'] = ssreg / sstot

    return results,p

# %%
y_train_list_electron=y_train_list[element_shapes[0]:]
y_train_list_gamma=y_train_list[:element_shapes[0]]

for i,y in enumerate(y_pred_all):
    y=y.squeeze()

    y_electron=y[element_shapes[0]:]
    y_gamma=y[:element_shapes[0]]

    x_aux=np.linspace(-0.1,3.2,50)

    results,model_poly=polyfit(y_train_list,y,1)
    results_electron,model_poly_electron=polyfit(y_train_list_electron,y_electron,1)
    results_gamma,model_poly_gamma=polyfit(y_train_list_gamma,y_gamma,1)
    
    rmse=np.sqrt(np.mean((y-y_train_list)**2))
    rmse_gamma=np.sqrt(np.mean((y_gamma-y_train_list_gamma)**2))
    rmse_electron=np.sqrt(np.mean((y_electron-y_train_list_electron)**2))

    #y_poly_fit=model_poly(x_aux)
    fig=plt.figure(figsize=(7,7))
    ax=plt.gca()
    plt.plot(y_train_list_gamma,y_gamma,".",color="purple",label=f"gamma (#{element_shapes[0]})",alpha=0.8)
    plt.plot(y_train_list_electron,y_electron,".",color="green",label=f"electron (#{element_shapes[1]})",alpha=0.2)

    plt.plot(np.arange(7),"--r",label="Slope=1")
    sns.regplot(data=pd.DataFrame({"X":y_train_list,"Y":y}),x="X",y="Y",ax=ax,line_kws={'color': 'orange'},label=f"Regresion global (slope={results['polynomial'][0]:.3f})",scatter=False)
    sns.regplot(data=pd.DataFrame({"X":y_train_list_gamma,"Y":y_gamma}),x="X",y="Y",ax=ax,line_kws={'color': 'purple'},label=f"Regresion gamma (slope={results_gamma['polynomial'][0]:.3f})",scatter=False)
    sns.regplot(data=pd.DataFrame({"X":y_train_list_electron,"Y":y_electron}),x="X",y="Y",ax=ax,line_kws={'color': 'green'},label=f"Regresion electron (slope={results_electron['polynomial'][0]:.3f})",scatter=False)

    plt.xlim(0,3.2)
    plt.ylim(0,3.5)
    plt.xlabel("y_true (TeV)")
    plt.ylabel("y_pred (TeV)")
    plt.legend()
    plt.title(f"Arquitectura {i}; \n R2={results['determination']:.2f}, RMSE={rmse:.3f} \n R2_electron={results_electron['determination']:.2f}, RMSE_electron={rmse_electron:.3f} \n R2_gamma={results_gamma['determination']:.2f}, RMSE_gamma={rmse_gamma:.3f}")
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/results/009_correlation_energy_filter_{i}_v2.png")
    plt.clf()
    plt.close(fig)
    #plot de la distribucion de errores
    fig2=plt.figure(figsize=(8,4))
    res=y_train_list-y
    res_gamma=y_train_list_electron-y_electron
    res_electron=y_train_list_gamma-y_gamma
    plt.hist(res,bins=70,label="Residuos global",density=True,color="orange",alpha=0.7)
    plt.hist(res_electron,bins=70,label="Residuos electron",density=True,color="green",alpha=0.4)
    plt.hist(res_gamma,bins=70,label="Residuos gamma",density=True,color="purple",alpha=0.3)
    plt.legend()
    plt.title(f"Arquitectura {i}")# ; std: {np.std(y_train_list-y)}")
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/results/009_resid_hist_energy_filter_{i}_v2.png")
    plt.clf()
    plt.close(fig2)


