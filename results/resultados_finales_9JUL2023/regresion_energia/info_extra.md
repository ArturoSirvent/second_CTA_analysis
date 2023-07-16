# Resultados Finales 9JUL2023  

## Regresión energías  (Solo se ha hecho con 4 telescopios, y para todo el espectro hasta 5 TeV)   
La regresión de las energías mediante una red neuronal convolucional se hizo con los siguientes hiperparámetros y condiciones:  
- CNN arch: Filtros (`[[32,64],[64,128],[128,64],[32,16]]`) + Dense (`[20,5]`)   
- lr: 1e-4   
- loss: mse   
- Inputs: 4 tels de 55 x 93 pixels   
- GPU: Las GPUs del cluster que me dio acceso Alberto.  (`NVIDIA GeForce RTX 2080 Ti`)  
- Calculo MeanAbsoluteError adicional, pero no se usa para nada mas.  
- 70 épocas.   
- Batchsize: 64  
- 20 porciento de los datos de entrenamiento son para validación.   
- Se usan las últimas 300 RUNS de cada elemento (gamma y electron). Recordemos que cada run tiene un número variable de eventos/detecciones. Debemos de tomar los eventos comunes a los 4 telescopios, porque es de los que tenemos información. Para ello, se escogen eventos comunes, y el número de eventos se iguala al menor, para balancear los ejemplos de diferentes clases. Es decir, que si tenemos 4k eventos de electron, disponibles para los 4 telescopios, y 9k eventos de gamma. Tomaremos 4k de gamma y los 4k de electron. Los número concretos son:       
Nº de runs (gamma, electron) usadas consideradas para entrenamiento -> [75, 276]  
Nº de runs (gamma, electron) usadas consideradas para test ->[20, 70]  
Nº Eventos electron para train -> 69096  
Nº Eventos gamma para train -> 93461    
Nº Eventos electron para test -> 17520    
Nº Eventos gamma para train -> 23744    
**Para train, tomamos el mismo número de eventos de ambos (si de uno hay menos, tomamos esa cantidad para el resto), En TEST por el contrario, no hacemos eso, aprovechamos todos los datos disponibles para hacer el test.**    

Los datos de la gráfica de las losses de train de la energia estan en `Second_CTA_analysis/notebooks/results_UGR/energias/test_2_300.h5`.  
Los datos de aplicar el modelo a los datos de test estan en :  `second_CTA_analysis/notebooks/results_UGR/energias/test-pred_energy_2_300.npy`.   



## Clasificación de elementos   
La clasificacion la tenemos para todo el rango de energias, y tanto para 1 como para 4 tels. Para 1 tel solo tenemos la clasificación en 3 clases, para 4 tels además de eso, tenemos la clasificación en las 7 clases (para mostrar la confusión entre adrones).  
- Arquitectura igual que para la regresión -> CNN arch: Filtros (`[[32,64],[64,128],[128,64],[32,16]]`) + Dense (`[20,5]`)     
- Loss usada **categorical_crossentropy**, con la correspondiente softmax en la ultima capa del clasificador.    
- lr: 1e-4  
- Métricas calculadas: acc, auc y MSE.   
- 40 epocas de entrenamiento.  
- Batchsize: 64  
- 20 porciento de los datos de entrenamiento son para validación.    
- Se usan las últimas 300 RUNS de cada elemento (gamma y electron). Recordemos que cada run tiene un número variable de eventos/detecciones. Debemos de tomar los eventos comunes a los 4 telescopios, porque es de los que tenemos información. Para ello, se escogen eventos comunes, y el número de eventos se iguala al menor, para balancear los ejemplos de diferentes clases. Es decir, que si tenemos 4k eventos de electron, disponibles para los 4 telescopios, y 9k eventos de gamma. Tomaremos 4k de gamma y los 4k de electron. Los número concretos son:       
Nº de runs ('gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon') usadas consideradas para entrenamiento -> [75, 278, 343, 255, 347, 318, 287]  
Para train:    
electron 17945  
gamma 87945  
helium 65102  
iron 40693  
nitrogen 70202  
proton 51281  
silicon 68890  

Nº de runs ('gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon') usadas consideradas para test -> [20, 68, 87, 65, 87, 82, 73] 
Para test:   
electron 3952  
gamma 22670  
helium 16826  
iron 10648  
nitrogen 17680  
proton 13208  
silicon 17316  

