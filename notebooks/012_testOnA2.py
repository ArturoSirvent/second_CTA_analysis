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
from numba import cuda
import gc

#propias
import unzipdata_and_first_treatments as manipulate
import loaddata4use
import model_creation_functions as models

PATH_TXT=f"{BASE_DIR}/data_full/combined/SimTelArray_2022_05"

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
    
df_clean=df_final.groupby(["elemento","run"]).apply(lambda x: x if len(list(x["tel"]))==4 else print(list(x["tel"]))).dropna().reset_index(drop=True)

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

df_common_events=df_clean.groupby(["elemento","run"]).apply(f_get_comon_events).droplevel(2).reset_index()#.to_frame()
eventos_number=df_common_events.groupby(["elemento","run"]).size().to_frame().reset_index()
eventos_number=eventos_number.rename(columns={0:"n"})
eventos_number_rand=eventos_number.groupby("elemento").apply(lambda x : x.sample(frac=1)).reset_index(drop=True)
eventos_number_rand["percent"]=eventos_number_rand.groupby("elemento")["n"].transform(lambda x : x/x.sum())
eventos_number_rand["cumsum"]=eventos_number_rand.groupby("elemento")["percent"].transform(lambda x : x.cumsum())
eventos_number_rand.loc[eventos_number_rand["cumsum"]<0.80,"mode"]="Train"
eventos_number_rand["mode"]=eventos_number_rand["mode"].fillna("Test")
df_lista_runs=eventos_number_rand.groupby(["elemento","mode"]).apply(lambda x : list(x["run"])).to_frame(name="list_runs").reset_index()

elementos=['gamma', 'electron', 'proton', 'helium', 'nitrogen', 'silicon', 'iron']
train_runs=df_lista_runs.loc[df_lista_runs["mode"]=="Train"].set_index("elemento")
train_runs=train_runs.loc[elementos].reset_index()
train_runs_list=list(train_runs["list_runs"].to_numpy())
test_runs=df_lista_runs.loc[df_lista_runs["mode"]=="Test"].set_index("elemento")
test_runs=test_runs.loc[elementos].reset_index()
test_runs_list=list(test_runs["list_runs"].to_numpy())

PATH_npy=f"{BASE_DIR}/data_full/combined/elementos_npy"

aux_list=[i[:2] for i in train_runs_list]

x_train_list,x_test_list,y_train_list,y_test_list=loaddata4use.load_dataset_completo(PATH_npy,labels_asign=[0,1,2,2,2,2,2],elements=elementos,
                                                                                    main_list_runs=aux_list,pre_name_folders="npy_",telescopes=[1,2,3,4],
                                                                                    test_size=0.1,same_quant="all",verbose=True,fill=True,categorical=True)

del test_runs,df_lista_runs,eventos_number,df_common_events, df_clean, df_final
def cambiar_ejes_lista(lista):
    for i,j in enumerate(lista):
        lista[i]=np.swapaxes(j,1,2)
    return lista


x_train_list=cambiar_ejes_lista(x_train_list)
x_test_list=cambiar_ejes_lista(x_test_list)

modelo=models.model_multi_tel(classes=3,filtros=[[12,16],[16]],last_dense=[20,5])
modelo.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["acc","AUC","mean_squared_error"])

device = cuda.get_current_device()
gc.collect()
hist=modelo.fit(x=x_train_list,y=y_train_list,epochs=2, validation_data=(x_test_list,y_test_list),batch_size=64)            
