# Fuentes y versión de los datos

`reclamos.zip` es la descarga oficial conservada para reproducir esta entrega. `metadata.txt` es la descripción de campos publicada. El manifiesto `fuentes.json` registra URLs, tamaño, SHA-256 y fecha de descarga.

El CSV descomprimido supera el límite habitual de GitHub y no se versiona. El notebook lo extrae del ZIP si no existe y verifica el hash del ZIP antes de leerlo. No hace falta descargar los datos otra vez para ejecutar.

La descarga fue realizada el 22 de octubre de 2026. El archivo público se sobrescribe periódicamente: una nueva descarga puede diferir de la versión analizada. `descargar_datos.py` permite recuperar las fuentes oficiales, pero no debe sobrescribirse el manifiesto de esta versión para comparar resultados históricos.

El catálogo describe coordenadas UTM, pero este CSV contiene `LATITUD` y `LONGITUD` en grados. El contraste se documenta en el notebook. El estado Finalizado es un cierre de gestión y no confirma por sí solo reparación física.

Fuente: https://catalogodatos.gub.uy/dataset/reclamos-registrados-en-el-sistema-unico-de-reclamos-de-la-intendencia-de-montevideo

Licencia indicada por el catálogo: Licencia de Datos Abiertos de Uruguay. El análisis no constituye un diagnóstico sobre la gestión municipal.
