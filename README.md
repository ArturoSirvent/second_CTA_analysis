# Estructuración del repositorio

```
.
├── automat
│   ├── logs
│   │   └── old
│   └── scripts
├── config
├── datos
│   ├── elementos
│   │   ├── extract_electron
│   │   ├── extract_gamma
│   │   ├── extract_helium
│   │   ├── extract_iron
│   │   ├── extract_nitrogen
│   │   ├── extract_proton
│   │   └── extract_silicon
│   ├── elementos_npy
│   │   ├── npy_electron
│   │   ├── npy_gamma
│   │   ├── npy_helium
│   │   ├── npy_iron
│   │   ├── npy_nitrogen
│   │   ├── npy_proton
│   │   └── npy_silicon
│   ├── elementos_npy_test
│   │   ├── npy_electron
│   │   ├── npy_gamma
│   │   ├── npy_helium
│   │   ├── npy_iron
│   │   ├── npy_nitrogen
│   │   ├── npy_proton
│   │   └── npy_silicon
│   └── zips
├── doc
├── modelos
│   ├── old
│   └── performances
│       └── old
├── notebooks
│   └── old
├── pipeline
│   └── sripts
├── results
├── src
│   └── CTA-data-analisis-library
│       ├── imgs
│       └── __pycache__
└── tmp_execution
    └── DONE

```

La idea era tenerlo todo lo mejor clasificado y segmentado posible, pero hay algunas cosas que al final no pude organizar todo lo que me gustaría.

El flujo de trabajo resumidamente era:  
1. Hago un Notebook (notebooks) con el codigo que quiero ejecutar, pero sin poner muchas épocas, sin cargar muchos datos, para que no tarde en ejecutar.
2. Una vez visto que no hay errores, lo paso a `.py` , y lo copio en la carpeta  __tmp_execution__. Esto lo hacía porque si lo dejo en la misma carpeta __notebooks__, y volvia una semana después, ya no me acordaba que se habia quedado ejecutando. Y una vez termine, lo muevo a __tmp_execution/DONE__.
3. Con el código a ejecutar en __tmp_execution__, el encargado de ejecutarlo, es el script de __automat/sripts__, que comparten ID para poder identificarlo rapido. Este `.sh` tiene las instrucciones de `slurm` para ejecutarlo. Lo ejecutaba con `sbatch script.sh`, y el log se guardaba en la misma carpeta (porque sino me salia un error). Pero los logs de `slurm` son archivos muy muy largos, y no los he subido. EN su lugar, tenemos un log que creaba yo en run time, y estan en __automat/logs__.
4. Por último, al ejecutarse, se rellenaban tanto el log `.out` de slurm, como el log propio en la carpeta __logs__. Y se guardaban **todos** los modelos y _performances_ del entrenamiento en la carpeta __results__. Sí, lo ideal hubiera sido poner un metodo de guardar solo el mejor modelo, pero no sabía hacerlo en el momento.



Sobre los datos: 
- La carpeta __datos/zips__ tiene los datos originales comprimidos.   
- Luego, encontramos los archivos principales descomprimidos en la carpeta **datos/elementos**. Una carpeta por elemento. En estas carpetas esta el `.dt` que tiene la captura del telescopio, y los `.txt` que tiene info sobre la simulación.   
- Para hacer más eficiente la carga de datos, se procesó y guardo todo en `.npy` .Y tenemos __datos/elementos_npy__ con los datos de entrenamiento, y __datos/elementos_npy_test__ con los de test. Dentro de estas carpetas hay dos tipos de archivos, los que tiene la imagen capturada por el telescopio, y los que indican el ID del evento. Esto de los IDS es muy importante, porque si queremos cargar elementos para los telecopios: 4 , 5, 22, 61, 123. Solo podremos usar eventos en los que se vean involucrados TODOS los telecopios, pero no todos los telescopios registran todos los eventos, muchísimos estan vacios, sobretodo para los relecopios pequeños y lejanos. Estos archivos con IDs se usan para eso, para saber que eventos cargar, antes de cargar las imagenes, que es lo costoso. 


__Notas extra:__ 
- Hay algunos scripts que sobran como, `scripts/long_exec.py`, esto fueron para hacer ciertas pruebas del cluster. 
- La carpeta **pipeline** no se ha usado.
- Hay carpetas __old__ a modo de archivo legacy.
- En la carpeta __src__ tenemos la libreria/repositorio de mi TFG, donde estan las funciones necesarias para el análisis etc.  
- __doc__ tiene anotaciones que iba haciendo, pero esta bastante incompleto.





Aclaración sobre el funcionamiento de la librerias/repositorio de mi TFG: 
- Hay muchas muchísimas funciones que a la hora de la verdad no hace falta usar. Pues o bien las implementan otras fucniones más grandes, o bien fueron diseñadas para algo muy concreto que no hace falta casi nunca.
- Lo básico que se hace es:
    1. Cargar los datos: Cargaremos X runs, para los telescopios que indiquemos, para los elementos que indiquemos, y los etiquetaremos según queramos (si dos elementos tienen la misma etiqueta, se consideran de la misma clase), y tendremos también una forma de elegir cargar todos los elementos disponibles o igualar el número de eventos de cada clase etc...
    2. Creamos el modelo según el numero de inputs, su forma, la arquitectura de la parte convolucional, el numero de outputs, la arquitectura deseada de la parte final, y el uso de un primer modelo preentrenado tipo autoencoder.
    3. Se entrenan los modelos.  
    4. Evaluación: hay algunas funciones para crear la matriz de confusion etc.


Nota: Tanto la carga de los datos, como la creacion de los modelos, tiene su version clasificación de elementos, como regresion de energías. Y en el caso de regresión de energias, en la carga de modelos se puede escoger el rango de energías deseado. 


_Escrito a 20 abril de 2023._

_Arturo Sirvent_

