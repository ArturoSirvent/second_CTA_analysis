# %% [markdown]
# Vamos a poder seleccionar para el entrenamiento sobre todas las características que queramos.  

# %%
import sys 
BASE_DIR="/home/arturosf/Documentos/repos/second_CTA_analysis"
sys.path.append(f"{BASE_DIR}/src/CTA-data-analisis-library/")
import numpy as np 
import matplotlib.pyplot as plt
import os 
import re 
import glob 
import tensorflow as tf 
import seaborn as sns
import pandas as pd

#propias
import unzipdata_and_first_treatments as manipulate
import loaddata4use
import model_creation_functions as models

# %%
PATH_TXT=f"{BASE_DIR}/data_full/combined/SimTelArray_2022_05"



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
elementos=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']
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
elementos=['gamma', 'electron', 'proton', 'helium', 'nitrogen', 'silicon', 'iron']

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

df_lista_runs.to_csv("runs_train_test.csv")

# %%
elementos=['gamma', 'electron', 'proton', 'helium', 'nitrogen', 'silicon', 'iron']
train_runs=df_lista_runs.loc[df_lista_runs["mode"]=="Train"].set_index("elemento")
train_runs=train_runs.loc[elementos].reset_index()
train_runs_list=list(train_runs["list_runs"].to_numpy())


# %%
test_runs=df_lista_runs.loc[df_lista_runs["mode"]=="Test"].set_index("elemento")
test_runs=test_runs.loc[elementos].reset_index()
test_runs_list=list(test_runs["list_runs"].to_numpy())

# %%
PATH_npy=f"{BASE_DIR}/data_full/combined/elementos_npy"
# %%
#aux_list=[i[:2] for i in train_runs_list]

# %%
x_train_list,x_test_list,y_train_list,y_test_list=loaddata4use.load_dataset_completo(PATH_npy,labels_asign=[0,1,2,2,2,2,2],elements=elementos,
                                                                                    main_list_runs=train_runs_list,pre_name_folders="npy_",telescopes=[1,2,3,4],
                                                                                    test_size=0.01,same_quant="approx",verbose=True,fill=True,categorical=True)


print([i.shape for i in x_train_list],[ i.shape for i in x_test_list],y_train_list.shape,y_test_list.shape)

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
x_train_list=cambiar_ejes_lista(x_train_list)
x_test_list=cambiar_ejes_lista(x_test_list)


with tf.device("CPU"):
    x_train_tensor_list=[tf.convert_to_tensor(i) for i in x_train_list]
    x_test_tensor_list=[tf.convert_to_tensor(i) for i in x_test_list]
    y_train_tensor=tf.convert_to_tensor(y_train_list)
    y_test_tensor=tf.convert_to_tensor(y_test_list)

# %%
modelo=models.model_multi_tel(classes=3,filtros=[[32,64],[64,128],[128,64],[32,16]],len_inputs=4,last_dense=[20,5])
modelo.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["acc","AUC","mean_squared_error"])

# %%
from numba import cuda
import gc
import pickle
device = cuda.get_current_device()


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
hist=modelo.fit(x=x_train_tensor_list,y=y_train_tensor,epochs=30, validation_data=(x_test_tensor_list,y_test_tensor),batch_size=128)#,callbacks=[Print_gpu_usage()])     

modelo.save(f"test.h5")
with open(f"hist.pickle","wb") as pick:
    pickle.dump(hist.history,pick)
