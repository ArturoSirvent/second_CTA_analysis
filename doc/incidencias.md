## Problemas con los datos: 

* Tenemos un problema con algunos archivos faltantes, como por ejemplo, para el telecopio 3 y 4 no tenemos el archivo .dt de la run 2

## Problemas con la memoria y el tiempo de ejecución:  

* Tenemos el problema de tiempo de ejecución, parecía que el ordenador cancelaba la ejecución cada cierto tiempo, pero no es así. Aunque algo pasa porque se paraba la ejecución.  
**El problema del tiempo de ejecución estaba siendo que los output del nohup se graban en buffer por lotes o algo así, pero si lo ponemos al ejecutarlos con pytohn -u, asi se graba constantemente.**  
* También teníamos un error con la memoría en tiempo de ejecución y parece que lo he arreglado borrando memoria cache de keras y usando la libreria de garbage collection.
