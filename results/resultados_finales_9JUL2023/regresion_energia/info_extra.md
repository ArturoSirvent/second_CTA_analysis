# Resultados Finales 9JUL2023  

## Regresión energías  (Solo se ha hecho con 4 telescopios)  
La regresión de las energías mediante una red neuronal convolucional se hizo con los siguientes hiperparámetros y condiciones:  
- CNN arch: Filtros (`[[32,64],[64,128],[128,64],[32,16]]`) + Dense (`[20,5]`)   
- lr: 1e-4   
- loss: mse   
- Inputs: 4 tels de 55 x 93 pixels   
- GPU: Las GPUs del cluster que me dio acceso Alberto.  (`NVIDIA GeForce RTX 2080 Ti`)  
- Calculo MeanAbsoluteError adicional, pero no se usa para nada mas.  
- 70 épocas.   
- Batchsize: 64   
- Se usan las últimas 300 RUNS de cada elemento (gamma y electron). Recordemos que cada run tiene un número variable de eventos/detecciones. Debemos de tomar los eventos comunes a los 4 telescopios, porque es de los que tenemos información. Para ello, se escogen eventos comunes, y el número de eventos se iguala al menor, para balancear los ejemplos de diferentes clases. Es decir, que si tenemos 4k eventos de electron, disponibles para los 4 telescopios, y 9k eventos de gamma. Tomaremos 4k de gamma y los 4k de electron. Los número concretos son:       
Nº de runs (gamma, electron) usadas consideradas para entrenamiento -> [75, 276]  
Nº de runs (gamma, electron) usadas consideradas para test ->[20, 70]  
Nº Eventos electron para train -> 69096  
Nº Eventos gamma para train -> 93461    
Nº Eventos electron para test -> 17520    
Nº Eventos gamma para train -> 23744    
**Para train, tomamos el mismo número de eventos de ambos (si de uno hay menos, tomamos esa cantidad para el resto), En TEST por el contrario, no hacemos eso, aprovechamos todos los datos disponibles para hacer el test.**    

Los datos de la gráfica de las losses de train de la energia estan en `Second_CTA_analysis/notebooks/results_UGR/energias/test_2_300.h5`.  
Los datos de aplicar el modelo a los datos de test estan en :  