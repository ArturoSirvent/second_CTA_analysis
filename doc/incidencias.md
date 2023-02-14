## Problemas con los datos: 

* Tenemos un problema con algunos archivos faltantes, como por ejemplo, para el telecopio 3 y 4 no tenemos el archivo .dt de la run 2

## Problemas con la memoria y el tiempo de ejecución:  

* Tenemos el problema de tiempo de ejecución, parecía que el ordenador cancelaba la ejecución cada cierto tiempo, pero no es así. Aunque algo pasa porque se paraba la ejecución.  
**El problema del tiempo de ejecución estaba siendo que los output del nohup se graban en buffer por lotes o algo así, pero si lo ponemos al ejecutarlos con pytohn -u, asi se graba constantemente.**  
* También teníamos un error con la memoría en tiempo de ejecución y parece que lo he arreglado borrando memoria cache de keras y usando la libreria de garbage collection.

* La informacion sobre uso de memoria en cpu que estamos displayeando con lo de tf solo refiere (creo) a uso del modelo, aunque no tiene mucho sentido algunas lecturas de la informacion que ocupa, siendo en los primeros boostraps mucho y luego menos.

* A pasado algo raro, en el tercer filtro propuesto para la energia, se ha parado en la primera epoca del primer boostrap, probablemente ha colapsado el proceso por uso de memoria, pero no se como comprobarlo. Es verdad que la red esa era la más grande de las tres, alomejor demasiados parámetros.


## He hecho unos modelos

Y bien, pero para el caso de la regresión de la energía falla mucho para mas de 1 TeV, entonces voy a hacer un modelo que solo experimente con valores mayores que 1TeV.