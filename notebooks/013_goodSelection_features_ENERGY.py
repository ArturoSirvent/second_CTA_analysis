# %% [markdown]
# Vamos a poder seleccionar para el entrenamiento sobre todas las características que queramos.  

# %%
import sys 
BASE_DIR="/home/asirvent/second_CTA_analysis"
sys.path.append(f"{BASE_DIR}/src/CTA-data-analisis-library/")
import numpy as np 
import matplotlib.pyplot as plt
import os 
import re 
import glob 
import tensorflow as tf 
import seaborn as sns
import pandas as pd
import gc
import pickle
#propias
import unzipdata_and_first_treatments as manipulate
import loaddata4use
import model_creation_functions as models

# %%
PATH_TXT=f"/home/asirvent/SimTelArray_2022_05"
RESULTS_DIR="/home/asirvent/second_CTA_analysis/notebooks/results_UGR/energias"



# %% [markdown]
# Tengo todas las runs y todos los eventos con sus respectivas runs. Voy a hacer lo siguiente.  
# - Tomaré todos los eventos disponibles para una distribución uniforme de energias ,y haré tanto la clasificacion como la regresión de energías. Con los mismo RUNS para el train y luego test unos diferentes que para train, pero los mismos entre clasificación y regresion.  
# 

# %%
archivo_min_pares = [1, 1, 1, 1, 1, 1, 1]
archivo_min_indices_pares = [91, 361, 421, 309, 387, 347, 417]
archivo_max_pares = [90, 360, 420, 308, 386, 346, 416]
archivo_max_indices_pares = [100, 370, 430, 320, 400, 360, 434]



# %%
# voy a cargar toda la info de los txt para cada elemento 
elementos=['gamma', 'electron']
aux2=[]
for i , elemento in enumerate(elementos):
    #dict_master[elemento]={}
    list_aux=glob.glob(f"{PATH_TXT}/extract_{elemento}/*.txt")
    list_aux=sorted(list_aux)

    for dir_aux in list_aux:
        run_aux=int(re.findall("(\d{3})\.txt",dir_aux)[0])
        tel_aux=int(re.findall("tel_(\d{1})_run",dir_aux)[0])
        #La idea es tener un diccionario enorme de todo lo que podemos necesitar para seleccionar el rango de todo y las distribuciones.  
        # Esto lo logramos con los datos de extract  
        array_aux=loaddata4use.extract_info_txt(dir_aux)
        if array_aux.size>0:
            aux2.append({"elemento":elemento,"run":run_aux,"tel":tel_aux,"energia":array_aux[:,2],"eventos":array_aux[:,0].astype(int)})
        else:
            aux2.append({"elemento":elemento,"run":run_aux,"tel":tel_aux,"energia":np.nan,"eventos":np.nan})
df_final=pd.DataFrame.from_dict(aux2)
    

# %%
df_clean=df_final.groupby(["elemento","run"]).apply(lambda x: x if len(list(x["tel"]))==4 else print(list(x["tel"]))).dropna().reset_index(drop=True)


# %%
def f_get_comon_events(df):
    all_eventos=np.concatenate(df["eventos"].to_numpy())
    event,count=np.unique(all_eventos,return_counts=True)
    all_energias=np.concatenate(df["energia"].to_numpy())
    energias_eventos_comunes=[]
    for i in event[count==4]:
        indx=np.argwhere(i==all_eventos)
        if np.unique(all_energias[indx]).size!=1:
            print("Algo ha pasado con el evento i, no coinciden las energias")
        energias_eventos_comunes.append(all_energias[indx[0]][0])
    return pd.DataFrame({"eventos":event[count==4],"energia":energias_eventos_comunes})

# %%
df_common_events=df_clean.groupby(["elemento","run"]).apply(f_get_comon_events).droplevel(2).reset_index()#.to_frame()



# %%
# vamos a hacerlo para los eventos antes y despues del cambio ese de la simulacion  

#estos siguen este orden -> ['gamma', 'electron', 'proton', 'helium', 'nitrogen', 'silicon', 'iron']
archivo_min_pares = [1, 1, 1, 1, 1, 1, 1]
archivo_min_indices_pares = [91, 361, 421, 309, 387, 347, 417]
archivo_max_pares = [90, 360, 420, 308, 386, 346, 416]
archivo_max_indices_pares = [100, 370, 430, 320, 400, 360, 434]

aux_df=[]
for j,i in enumerate(elementos):
    element_aux_df=df_common_events.loc[df_common_events["elemento"]==i]
    aux_df.append(element_aux_df.loc[(element_aux_df["run"]>=archivo_min_pares[j])&(element_aux_df["run"]<=archivo_max_pares[j])])

# %%
aux_df=[]
for j,i in enumerate(elementos):
    element_aux_df=df_common_events.loc[df_common_events["elemento"]==i]
    aux_df.append(element_aux_df.loc[(element_aux_df["run"]>=archivo_min_indices_pares[j])&(element_aux_df["run"]<=archivo_max_indices_pares[j])])
df_common_events_aux1=pd.concat(aux_df)


# %%
eventos_number=df_common_events.groupby(["elemento","run"]).size().to_frame().reset_index()
eventos_number=eventos_number.rename(columns={0:"n"})

# %%
#vamos a seleccionar unas runs para el train y otras para el test   
eventos_number_rand=eventos_number.groupby("elemento").apply(lambda x : x.sample(frac=1)).reset_index(drop=True)

# %%
eventos_number_rand["percent"]=eventos_number_rand.groupby("elemento")["n"].transform(lambda x : x/x.sum())

# %%
eventos_number_rand["cumsum"]=eventos_number_rand.groupby("elemento")["percent"].transform(lambda x : x.cumsum())

# %%
#ahora para cada elemento, vamos a guardar los runs que contienen el 80% de los datos
eventos_number_rand.loc[eventos_number_rand["cumsum"]<0.80,"mode"]="Train"
eventos_number_rand["mode"]=eventos_number_rand["mode"].fillna("Test")


# %%
df_lista_runs=eventos_number_rand.groupby(["elemento","mode"]).apply(lambda x : list(x["run"])).to_frame(name="list_runs").reset_index()

df_lista_runs.to_csv(f"{RESULTS_DIR}/runs_train_test_energy_2.csv")

# %%
train_runs=df_lista_runs.loc[df_lista_runs["mode"]=="Train"].set_index("elemento")
train_runs=train_runs.loc[elementos].reset_index()
train_runs_list=list(train_runs["list_runs"].to_numpy())


# %%
test_runs=df_lista_runs.loc[df_lista_runs["mode"]=="Test"].set_index("elemento")
test_runs=test_runs.loc[elementos].reset_index()
test_runs_list=list(test_runs["list_runs"].to_numpy())

# %%
PATH_npy=f"{BASE_DIR}/data_full/elementos_npy"
# %%
aux_list=[i[-300:] for i in train_runs_list]

# %%
print("Antes de cargar")

x_train_list,x_test_list,y_train_list,y_test_list=loaddata4use.load_dataset_energy(PATH_npy,PATH_TXT,elementos=['gamma', 'electron'],main_list_runs=aux_list,telescopios=[1,2,3,4],test_size=0.1,same_quant="approx",verbose=True,fill=True,lower_energy_bound=0,upper_energy_bound=7)

print("Después de cargar")
gc.collect()
# try:
#     print([i.shape for i in x_train_list],[ i.shape for i in x_test_list],y_train_list.shape,y_test_list.shape)
# except:
#     print("error con el prinde las shapes")
# %%
# print(tf.test.gpu_device_name())
# print(tf.config.list_physical_devices('GPU') )
# print(tf.config.experimental.get_memory_usage('GPU:0'))
print(tf.config.experimental.get_memory_info('GPU:0'))

# %%
def cambiar_ejes_lista(lista):
    for i,j in enumerate(lista):
        lista[i]=np.swapaxes(j,1,2)
    return lista

# %%
print("Cambio de ejes")

x_train_list=cambiar_ejes_lista(x_train_list)
x_test_list=cambiar_ejes_lista(x_test_list)
print("Ejes cambiados")


with tf.device("CPU:0"):
    print("Convertir a tensores")
    x_train_tensor_list=tf.cast(tf.convert_to_tensor(x_train_list), tf.float32)#[ for i in x_train_list]
    print("1")
    del x_train_list
    gc.collect()
    x_test_tensor_list=tf.cast(tf.convert_to_tensor(x_test_list), tf.float32)#[ for i in x_test_list]
    del x_test_list
    gc.collect()
    print("2")
    y_train_tensor=tf.cast(tf.convert_to_tensor(y_train_list),tf.float32)
    del y_train_list
    gc.collect()
    y_test_tensor=tf.cast(tf.convert_to_tensor(y_test_list),tf.float32)
    del y_test_list 
    gc.collect()
print("Convertidos a ")
gc.collect()

print("Creando modelo")
# %%
arch=[[32,64],[64,128],[128,64],[32,16]]
modelo=models.model_multi_tel_energy(len_inputs=4,input_shapes=[(55,93,1)],filtros=arch,last_dense=[20,5]) #no compila
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss="mse",metrics=["mae","mape"])
print("Modelo creado")


# %%
class Print_gpu_usage(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print("\n",tf.config.experimental.get_memory_info('GPU:0'))  
    #gc.collect()     

  # def on_train_batch_end(self, batch, logs=None):
  #   print("\n",tf.config.experimental.get_memory_info('GPU:0'))       



# %%

#con mas datos cargados en memoria

#con mas datos cargados en memoria
print("Comienza entrenamiento")

hist=modelo.fit(x=x_train_tensor_list,y=y_train_tensor,epochs=70, validation_data=(x_test_tensor_list,y_test_tensor),batch_size=64)#,callbacks=[Print_gpu_usage()])     

print("Entrenado el modelo")

modelo.save(f"{RESULTS_DIR}/test_2_300.h5")
with open(f"{RESULTS_DIR}/hist_2_300.pickle","wb") as pick:
    pickle.dump(hist.history,pick)
print("FIN")
